#!/bin/bash
SBATCH --ntasks=10
SBATCH --job-name=RF_100
SBATCH --output=rf_%j.out
SBATCH --exclude=nodo17

# Copyright(c) 202 QiangLi
# All Rights Reserved.
# qiang.li@uv.es
# Distributed under the (new) BSD License.

 
module load Anaconda3
source activate your_virtual_environment
 
python -u  BioMulti_L_NL_Model_Sample.py
