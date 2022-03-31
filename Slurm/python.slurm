#!/bin/bash
#SBATCH --partition LADON-GPU
#SBATCH --ntasks=1
# SBATCH --cpus-per-task=2
#SBATCH --job-name=video_data
#SBATCH --gres=gpu:1
#SBATCH --output=video_data.out

module load Anaconda3
source activate py3_env_updates

python -u video_bio2tfrecord_32x32_val.py