from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import geffnet

# from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
#                         build_sampler, multiclass_nms)

from mmdet.structures.bbox import bbox2result, bbox2roi, bbox_mapping
from mmdet.models.task_modules import build_assigner, build_sampler
from mmengine.structures import InstanceData

# from .. import builder
# from ..registry import EFFICIENTPS
from mmdet.registry import MODELS

from .base import BaseDetector
from mmdet.ops.norm import norm_cfg
from mmdet.ops.roi_sampling import roi_sampling, invert_roi_bbx

import mmengine.logging

import time
import cv2
import numpy as np
import os

@MODELS.register_module()
class EfficientPS(BaseDetector):

    def __init__(self,
                backbone,
                rpn_head,
                bbox_roi_extractor,
                bbox_head,
                mask_roi_extractor,
                mask_head,
                semantic_head,
                train_cfg,
                test_cfg,
                neck=None,
                shared_head=None,
                pretrained=None,
                use_unc=False,
                out_dir_unc="./tmpDir"):
        assert backbone is not None
        assert rpn_head is not None
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert mask_roi_extractor is not None           
        assert mask_head is not None
        assert semantic_head is not None

        super(EfficientPS, self).__init__()

        self.eff_backbone_flag = False if 'efficient' not in backbone['type'] else True

        if self.eff_backbone_flag == False:
            self.backbone = MODELS.build(backbone) # builder.build_backbone(backbone)
        else:
            self.backbone = geffnet.create_model(backbone['type'], 
                                                pretrained=True, # if pretrained is not None else False,
                                                se=False, 
                                                act_layer=backbone['act_cfg']['type'],
                                                norm_layer=norm_cfg[backbone['norm_cfg']['type']][1],
            ) 

        self.unc = use_unc

        self.neck = MODELS.build(neck) # builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = MODELS.build(shared_head) # builder.build_shared_head(shared_head)

        rpn_head.update(dict(train_cfg=train_cfg['rpn']))

        self.rpn_head = MODELS.build(rpn_head) # builder.build_head(rpn_head)

        # self.rpn_head.train_cfg = train_cfg['rpn'] # new

        self.bbox_roi_extractor = MODELS.build (bbox_roi_extractor) # builder.build_roi_extractor(bbox_roi_extractor)
        
        self.bbox_head = MODELS.build(bbox_head) # builder.build_head(bbox_head)

        self.mask_roi_extractor = MODELS.build(mask_roi_extractor) # builder.build_roi_extractor(mask_roi_extractor)
                    
        self.share_roi_extractor = True
        self.mask_head = MODELS.build(mask_head) # builder.build_head(mask_head)

        self.semantic_head = MODELS.build(semantic_head) # builder.build_head(semantic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
        self.num_classes = semantic_head['num_classes']
        self.num_stuff = self.num_classes - bbox_head['num_classes'] + 1
        self.init_weights(pretrained=pretrained)
        self.out_dir = out_dir_unc
        os.makedirs(self.out_dir, exist_ok=True)



    def init_weights(self, pretrained=None):
        if self.eff_backbone_flag == False:
            self.backbone.init_weights(pretrained=pretrained)

        self.neck.init_weights()

        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)

        self.rpn_head.init_weights()
        self.bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()
        self.mask_head.init_weights()
        self.mask_roi_extractor.init_weights()
        self.semantic_head.init_weights() 

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        x = self.neck(x)
        return x
    def forward_dummy(self, img): #leave it for now
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(device=img.device)
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def parse_losses(self, losses):
        
        log_vars = {}
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum(value for key, value in log_vars.items() if 'loss' in key)
        log_vars['loss'] = loss
        return loss, log_vars
    
    def forward(self, inputs, data_samples, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(inputs, data_samples, **kwargs)
        else:
            return self.forward_test(inputs, data_samples, **kwargs)

    def train_step(self, data, optim_wrapper):
        losses = self.forward(**data, return_loss=True)
        loss, log_vars = self.parse_losses(losses)

        optim_wrapper.update_params(loss)
        
        outputs = dict(
            loss=loss, num_samples=len(data['inputs']))
        outputs.update(log_vars)
        
        return outputs

    def val_step(self, data):
        outputs = self.forward(**data, return_loss=False)
        return outputs

    def test_step(self, data):
        outputs = self.forward(**data, return_loss=False)
        return outputs

    def assign_result(self, x, proposal_list,
                    img, gt_bboxes, gt_labels, gt_bboxes_ignore):
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            pred_instances = proposal_list[i].new()
            pred_instances.priors = proposal_list[i].bboxes 

            gt_instances = InstanceData()
            gt_instances.bboxes = gt_bboxes[i]
            gt_instances.labels = gt_labels[i]

            gt_ignore = None
            if gt_bboxes_ignore[i] is not None:
                gt_ignore = InstanceData()
                gt_ignore.bboxes = gt_bboxes_ignore[i]

            assign_result = bbox_assigner.assign(pred_instances,
                                                 gt_instances,
                                                 gt_instances_ignore=gt_ignore)

            sampling_result = bbox_sampler.sample(
                    assign_result,
                    pred_instances,
                    gt_instances,
                    feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        return sampling_results

    def forward_train(self, inputs, data_samples, **kwargs):
        # print(inputs)
        
        if isinstance(inputs, list):
            if len(inputs) > 1: 
                inputs = torch.stack(inputs)
            else:
                inputs = inputs[0]
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        inputs = inputs.to(next(self.parameters()).device)
        device = inputs.device

        for ds in data_samples:
            ds.gt_instances.bboxes = ds.gt_instances.bboxes.to(device)
            ds.gt_instances.labels = ds.gt_instances.labels.to(device)
            # if hasattr(ds.gt_instances, 'masks'):
            #     if hasattr(ds.gt_instances.masks, 'to_tensor'):
            #         ds.gt_instances.masks = ds.gt_instances.masks.to_tensor(dtype=torch.uint8, device=device)
            if hasattr(ds.gt_instances, 'ignored') and ds.gt_instances.ignored is not None:
                ds.gt_instances.ignored = ds.gt_instances.ignored.to(device)
            if hasattr(ds, 'gt_sem_seg'):
                ds.gt_sem_seg.sem_seg = ds.gt_sem_seg.sem_seg.to(device)
        
        img_metas = [ds.metainfo for ds in data_samples]
        gt_bboxes = [ds.gt_instances.bboxes for ds in data_samples]
        gt_labels = [ds.gt_instances.labels for ds in data_samples]
        gt_masks = [ds.gt_instances.masks for ds in data_samples]
        gt_semantic_seg = torch.stack([ds.gt_sem_seg.sem_seg for ds in data_samples])
        gt_bboxes_ignore = [getattr(ds.gt_instances, 'ignored', None) for ds in data_samples]


        x = self.extract_feat(inputs)
        losses = dict()

        semantic_logits = self.semantic_head(x[:4])
        gt_semantic_seg = gt_semantic_seg.to(semantic_logits.device)
        loss_seg = self.semantic_head.loss(semantic_logits, gt_semantic_seg)
        losses.update(loss_seg)

        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                        self.train_cfg.rpn)
        
        # old
        # rpn_losses = self.rpn_head.loss(
        #         *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # import pdb
        # pdb.set_trace()

        # pad shape is missing
        for ds in data_samples:
            if 'pad_shape' not in ds.metainfo:
                ds.set_metainfo(dict(pad_shape=ds.metainfo['img_shape']))


        rpn_losses = self.rpn_head.loss(x, data_samples)

        losses.update(rpn_losses)

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_metas, proposal_cfg)

        # old
        # proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        # new
        # proposal_list = self.rpn_head.predict(x, data_samples, rescale=False)

        proposal_list = self.rpn_head.predict_by_feat(*rpn_outs,
                                                      batch_img_metas = img_metas,
                                                      cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn,),
                                                      rescale = False)

        sampling_results =  self.assign_result(x, proposal_list, inputs,
                            gt_bboxes, gt_labels, gt_bboxes_ignore)
    
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        labels, label_weights, bbox_targets, bbox_weights = self.bbox_head.get_targets(sampling_results,
                                                                                            # gt_bboxes, gt_labels, # old with get_target instead of get_targets
                                                                                                self.train_cfg.rcnn)
        
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                        pos_rois,
                                        labels,
                                        label_weights,
                                        bbox_targets,
                                        bbox_weights)
        losses.update(loss_bbox)


        
        mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
        if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)

        if mask_feats.shape[0] > 0:
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False, eval=None):
        os.makedirs(self.out_dir, exist_ok=True)

        x = self.extract_feat(img)
        semantic_logits = self.semantic_head(x[:4])
        result = []
        if semantic_logits.shape[0] == 1:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                    self.test_cfg.rpn)

            det_bboxes, det_labels = self.simple_test_bboxes(x, 
                img_metas, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        
            if eval is not None:
                    
                panoptic_mask, cat_ = self.simple_test_mask_(
                    x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale)
                result.append([panoptic_mask, cat_, img_metas])
        
            else:          
                bbox_results = bbox2result(det_bboxes, det_labels,
                                        self.bbox_head.num_classes)
                mask_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale)

                return bbox_results, mask_results
        else:
            for i in range(len(img_metas)):
                new_x = []
                for x_i in x:
                    new_x.append(x_i[i:i+1])
                proposal_list = self.simple_test_rpn(new_x, [img_metas[i]],
                                    self.test_cfg.rpn)

                assert eval is not None

                det_bboxes, det_labels = self.simple_test_bboxes(new_x, 
                    [img_metas[i]], proposal_list, self.test_cfg.rcnn, rescale=rescale)

                panoptic_mask, cat_ = self.simple_test_mask_(
                    new_x, [img_metas[i]], det_bboxes, det_labels, semantic_logits[i:i+1], rescale=rescale)

                result.append([panoptic_mask, cat_, [img_metas[i]]])

        return result

    def aug_test(self,):
        pass


    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_bboxes(self,
                    x,
                    img_metas,
                    proposals,
                    rcnn_test_cfg,
                    rescale=False):

        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test_mask(self,
                x,
                img_metas,
                det_bboxes,
                det_labels,
                semantic_logits, 
                rescale=False):

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)

            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                    det_labels,
                                                    self.test_cfg.rcnn,
                                                    ori_shape, scale_factor,
                                                    rescale)
        return segm_result

    def simple_test_mask_(self,
                x,
                img_metas,
                det_bboxes,
                det_labels,
                semantic_logits, 
                rescale=False):

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        ref_size = (np.int(np.round(ori_shape[0]*scale_factor)), 
                    np.int(np.round(ori_shape[1]*scale_factor)))
        semantic_logits = F.interpolate(semantic_logits, size=ref_size, 
                                mode="bilinear", align_corners=False)   
        sem_pred = torch.argmax(semantic_logits, dim=1)[0]

        if self.unc:
            alpha = (F.softplus(semantic_logits) +1)
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha/S
            sem_unc = semantic_logits.shape[1] / S
            sem_unc=sem_unc.view(sem_unc.shape[2], sem_unc.shape[3])
            sem_unc_prob = 1 - sem_unc


        else:
            probs = F.softmax(semantic_logits, dim =1)
            entropy = -1.0*torch.multiply(probs, torch.log(probs))
            entropy_unc = torch.sum(entropy, dim =1)
            entropy_unc = entropy_unc.squeeze(0)
            entropy_unc = entropy_unc/torch.log(torch.tensor(19.0))
            sem_unc_prob = entropy_unc

        sem_unc_prob_file_name = img_metas[0]['filename'].split("/")[-1].replace(".png","_uncMap")
        np.save(os.path.join(self.out_dir, sem_unc_prob_file_name),sem_unc_prob.detach().cpu().numpy(),allow_pickle=True,)
        





        panoptic_mask = torch.zeros_like(sem_pred, dtype=torch.long)
        cat = [255]
        if det_bboxes.shape[0] == 0:
            intermediate_logits = semantic_logits[0, :self.num_stuff] 
        elif self.unc:
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            confidence = det_bboxes[:,4]

            idx = torch.argsort(confidence, descending=True)
            bbx_inv = invert_roi_bbx(det_bboxes[:, :4], 
                    tuple(mask_pred.shape[2:]), ref_size)
            bbx_idx = torch.arange(0, det_bboxes.size(0), 
                    dtype=torch.long, device=det_bboxes.device)
            mask_pred = roi_sampling(mask_pred, bbx_inv, bbx_idx, 
                        ref_size, padding="zero")
            
            ML_A = mask_pred.new_zeros(mask_pred.shape[0], mask_pred.shape[-2], 
                                            mask_pred.shape[-1])
            ML_X = mask_pred.new_zeros(mask_pred.shape[0], mask_pred.shape[-2], 
                                            mask_pred.shape[-1])
            ML_B = ML_A.clone()  
            ML_Y = ML_X.clone()             
            occupied = torch.zeros_like(sem_pred, dtype=torch.bool)

            i =0 
            
            for id_i in idx:
                label_i = det_labels[id_i]
                label_i = det_labels[id_i] 
                mask_pred_i = mask_pred[id_i, label_i+1, :, :]
                mask_log =  mask_pred[id_i, :, :, :]
                alpha = F.softplus(mask_log) +1
                Sm = torch.sum(alpha, dim=0, keepdim=True)
                mask_unc = mask_pred.shape[1] / Sm
                mask_unc = mask_unc.view(mask_unc.shape[1],mask_unc.shape[2])
                mu = (alpha/Sm)
                mask_unc = (alpha.shape[0]/Sm).squeeze(0)

                mp = mu[label_i+1,:,:]

                mask_i = (mask_pred_i> self.test_cfg.rcnn.mask_thr_binary) 
                mask_i = mask_i.type(torch.bool)
                intersection = occupied & mask_i
                if intersection.float().sum() / mask_i.float().sum() > self.test_cfg.panoptic.overlap_thr:
                    continue
                mask_i = mask_i ^ intersection
                occupied += mask_i
                y0 = max(int(det_bboxes[id_i, 1] + 1), 0)
                y1 = min(int((det_bboxes[id_i, 3] - 1).round() + 1), ref_size[0])
                x0 = max(int(det_bboxes[id_i, 0] + 1), 0)
                x1 = min(int((det_bboxes[id_i, 2] - 1).round() + 1), ref_size[1]) 

                m = mask_pred_i[y0: y1, x0: x1]
                
                
                mask_unc[y0: y1, x0: x1] =  (mask_unc[y0: y1, x0: x1] - torch.min(mask_unc[y0: y1, x0: x1])) / (torch.max(sem_unc[y0: y1, x0: x1]) - torch.min(mask_unc[y0: y1, x0: x1]))
                
                
                ML_Y[i, y0: y1, x0: x1] = sem_unc[y0: y1, x0: x1]
                
                
                ML_X[i] = torch.min(mask_unc[y0: y1, x0: x1])+(((torch.max(sem_unc[y0: y1, x0: x1])-torch.min(mask_unc[y0: y1, x0: x1])) * (mask_unc-torch.min(mask_unc[y0: y1, x0: x1])))/ (torch.max(mask_unc[y0: y1, x0: x1]) - torch.min(mask_unc[y0: y1, x0: x1])))#*mask_i
                
                
                mask_unc = mask_unc[y0: y1, x0: x1]
                
                mu = mu[label_i+1,y0: y1, x0: x1]

                ML_A[i] = mp
                ML_B[i, y0: y1, x0: x1] = probs[0, label_i + self.num_stuff, y0: y1, x0: x1]
                cat.append(label_i.item() + self.num_stuff)
                i = i + 1 

            ML_A = ML_A[:i]
            ML_B = ML_B[:i]
            ML_X = ML_X[:i]
            ML_Y = ML_Y[:i]

            FU = (ML_X + ML_Y)/2
            FL = (ML_A + ML_B)/2
            
            last_uncs,_ = torch.min(ML_X, dim=0)
            orig_sem_unc = sem_unc.clone()
            inst_uncs = last_uncs.clone()
            
            
            sem_uncs = sem_unc.unsqueeze(0).repeat(self.num_stuff,1,1)
            
            intermediate_logits = torch.cat([probs[0, :self.num_stuff], FL], dim=0)
            intermediate_uncs = torch.cat([sem_uncs, FU], dim=0)
            
            cat = torch.tensor(cat, dtype=torch.long)
            intermediate_mask = torch.argmax(intermediate_logits, dim=0) + 1
            mask_prob,prob_loc = torch.max(intermediate_logits, dim=0)
            maybe_uncs = (torch.gather(intermediate_uncs, 0 , prob_loc.unsqueeze(0))).squeeze(0)
            intermediate_mask = intermediate_mask - self.num_stuff
            intermediate_mask[intermediate_mask <= 0] = 0 

            sem_unc_1 = sem_unc.clone()
            sem_unc[torch.where(prob_loc >10)] =-1

            last_uncs[sem_unc !=-1]=0
            sem_unc[sem_unc==-1]=0


            ## for proper eval ####
            ov_unc =1-(last_uncs + sem_unc)

            ### for visualization ####
            ov_unc = 1- maybe_uncs
    
            unique = torch.unique(intermediate_mask) 
            ignore_val = intermediate_mask.max().item() + 1
            ignore_arr = torch.ones((ignore_val,), dtype=unique.dtype, device=unique.device) * ignore_val
            total_unique = unique.shape[0]
            ignore_arr[unique] = torch.arange(total_unique).cuda(ignore_arr.device)  
            panoptic_mask = ignore_arr[intermediate_mask]
            panoptic_mask[intermediate_mask == ignore_val] = 0 

            cat_ = cat[unique].long()
            sem_pred[panoptic_mask > 0] = self.num_stuff
            sem_pred[sem_pred >= self.num_stuff] = self.num_stuff
            cls_stuff, area = torch.unique(sem_pred, return_counts=True)
            cls_stuff[area < self.test_cfg.panoptic.min_stuff_area] = self.num_stuff
            cls_stuff = cls_stuff[cls_stuff!=self.num_stuff]     
            tmp = torch.ones((self.num_stuff + 1,), dtype=cls_stuff.dtype, device=cls_stuff.device) * self.num_stuff
            tmp[cls_stuff] = torch.arange(cls_stuff.shape[0]).cuda(tmp.device)  
            new_sem_pred = tmp[sem_pred]
            cat_ = torch.cat((cat_, cls_stuff.cpu().long()), -1)   
            bool_mask = new_sem_pred != self.num_stuff   
            panoptic_mask[bool_mask] = new_sem_pred[bool_mask] + total_unique 

            
            np.save(os.path.join(self.out_dir,sem_unc_prob_file_name+"_pan"),ov_unc.detach().cpu().numpy(),allow_pickle=True)
            return panoptic_mask.cpu(), cat_.cpu()
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            confidence = det_bboxes[:,4]
            idx = torch.argsort(confidence, descending=True)
            bbx_inv = invert_roi_bbx(_bboxes[:, :4], 
                    tuple(mask_pred.shape[2:]), ref_size)
            bbx_idx = torch.arange(0, det_bboxes.size(0), 
                    dtype=torch.long, device=det_bboxes.device)
            
            mask_pred = roi_sampling(mask_pred, bbx_inv, bbx_idx, 
                        ref_size, padding="zero")
            ML_A = mask_pred.new_zeros(mask_pred.shape[0], mask_pred.shape[-2], 
                                            mask_pred.shape[-1])
            ML_B = ML_A.clone()             
            occupied = torch.zeros_like(sem_pred, dtype=torch.bool)
            i = 0 
            for id_i in idx:
                label_i = det_labels[id_i] 
                mask_pred_i = mask_pred[id_i, label_i+1, :, :]
                mask_i = (mask_pred_i.sigmoid() > self.test_cfg.rcnn.mask_thr_binary) 
                mask_i = mask_i.type(torch.bool)
                intersection = occupied & mask_i
                if intersection.float().sum() / mask_i.float().sum() > self.test_cfg.panoptic.overlap_thr:
                    continue

                mask_i = mask_i ^ intersection
                occupied += mask_i

                y0 = max(int(_bboxes[id_i, 1] + 1), 0)
                y1 = min(int((_bboxes[id_i, 3] - 1).round() + 1), ref_size[0])
                x0 = max(int(_bboxes[id_i, 0] + 1), 0)
                x1 = min(int((_bboxes[id_i, 2] - 1).round() + 1), ref_size[1])

                ML_A[i] = 4 * mask_pred_i
                ML_B[i, y0: y1, x0: x1] = semantic_logits[0, label_i + self.num_stuff, y0: y1, x0: x1]
                cat.append(label_i.item() + self.num_stuff)
                i = i + 1 

            ML_A = ML_A[:i]
            ML_B = ML_B[:i]
            FL = (ML_A.sigmoid() + ML_B.sigmoid())*(ML_A + ML_B)
            intermediate_logits = torch.cat([semantic_logits[0, :self.num_stuff], FL], dim=0)

        cat = torch.tensor(cat, dtype=torch.long)
        intermediate_mask = torch.argmax(F.softmax(intermediate_logits, dim=0), dim=0) + 1
        intermediate_mask = intermediate_mask - self.num_stuff
        intermediate_mask[intermediate_mask <= 0] = 0         
        unique = torch.unique(intermediate_mask) 
        ignore_val = intermediate_mask.max().item() + 1
        ignore_arr = torch.ones((ignore_val,), dtype=unique.dtype, device=unique.device) * ignore_val
        total_unique = unique.shape[0]
        ignore_arr[unique] = torch.arange(total_unique).cuda(ignore_arr.device)  
        panoptic_mask = ignore_arr[intermediate_mask]
        panoptic_mask[intermediate_mask == ignore_val] = 0 

        cat_ = cat[unique].long()
        sem_pred[panoptic_mask > 0] = self.num_stuff
        sem_pred[sem_pred >= self.num_stuff] = self.num_stuff
        cls_stuff, area = torch.unique(sem_pred, return_counts=True)
        cls_stuff[area < self.test_cfg.panoptic.min_stuff_area] = self.num_stuff
        cls_stuff = cls_stuff[cls_stuff!=self.num_stuff]     

        tmp = torch.ones((self.num_stuff + 1,), dtype=cls_stuff.dtype, device=cls_stuff.device) * self.num_stuff
        tmp[cls_stuff] = torch.arange(cls_stuff.shape[0]).cuda(tmp.device)  
        new_sem_pred = tmp[sem_pred]
        cat_ = torch.cat((cat_, cls_stuff.cpu().long()), -1)   
        bool_mask = new_sem_pred != self.num_stuff   
        panoptic_mask[bool_mask] = new_sem_pred[bool_mask] + total_unique 

        return panoptic_mask.cpu(), cat_.cpu()

    def forward_test(self, inputs, data_samples, **kwargs):

        # Ensure inputs is a list
        if not isinstance(inputs, list):
            inputs = [inputs,]

        # Extract metainfo
        img_metas = [sample.metainfo for sample in data_samples]
        
        if len(inputs) == 1:
            return self.simple_test(inputs[0].unsqueeze(0), [img_metas[0]], **kwargs)
        else:
            return [
                self.simple_test(inputs[i].unsqueeze(0), [img_metas[i]], **kwargs)
                for i in range(len(inputs))
            ]
