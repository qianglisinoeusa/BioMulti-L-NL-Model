#!/bin/bash
#SBATCH --partition LADON
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=nest
#SBATCH --error=nest.err
#SBATCH --output=nest.out
#SBATCH --gres=gpu:0
#SBATCH --output=nest.out
#SBATCH --array=0-31   
#SBATCH --time=99:99:00

module load Anaconda3
source activate cnn

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts -y

python -u train_alexnet_flexabile.py
