#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
# SBATCH --job-name=model_TID
# SBATCH --error=model_TID_err.err
# SBATCH --output=model_TID_out.out
# More options
# SBATCH --nodes=2
# SBATCH --mem=256M
# SBATCH --gres=gpu:1
# SBATCH --time=99:99:00
# Load MATLAB module
module load MATLAB
# Uncomment these lines to use the local /scratch filesystem
## Create scratch working folder
#SCRATCH=/scratch/$USER/$SLURM_JOB_ID
#mkdir -p $SCRATCH || exit $?
## Copy needed files
#cp $SLURM_SUBMIT_DIR/matlab_normal.m $SCRATCH
#cd $SCRATCH
# Run command: IMPORTANT, USE < OR IT WON'T WORK

matlab -nodisplay < transform_IPL_scenes_DN_WC_A_1_forw_linear.m

# Uncomment these lines if using the local /scratch filesystem
## Get results
#cp $SCRATCH/figure.png $SLURM_SUBMIT_DIR
#cp $SCRATCH/result.mat $SLURM_SUBMIT_DIR
## Remove scratch folder
#rm -rf $SCRATCH