#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=train_EvPSnet
#SBATCH --output=slurm/train_EvPSnet.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.schader@students.uni-mannheim.de

echo "Started at $(date)";

module load devel/cuda/11.8

python tools/train.py configs/EvPSNet_unc_singlegpu.py --work_dir work_dirs/checkpoints --validate

end=$('date +%s')
runtime=$((end-start))

echo Runtime: $runtime