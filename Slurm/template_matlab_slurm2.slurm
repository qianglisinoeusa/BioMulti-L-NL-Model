#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=Entropy
#SBATCH --error=KING_err.err
#SBATCH --output=KING_out.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qiang.li@uv.es 
#SBATCH --time=160:00:00

#XXXSBATCH --account=qiangli
#XXXSBATCH --nodes=1
#XXXSBATCH --mem=256M
#XXXSBATCH --gres=gpu:1


#Load MATLAB module
module load MATLAB/R2016a
# Uncomment these lines to use the local /scratch filesystem
## Create scratch working folder
#SCRATCH=/scratch/$USER/$SLURM_JOB_ID
#mkdir -p $SCRATCH || exit $?
## Copy needed files
#cp $SLURM_SUBMIT_DIR/matlab_normal.m $SCRATCH
#cd $SCRATCH
# Run command: IMPORTANT, USE < OR IT WON'T WORK

matlab -nodesktop -nosplash -nodisplay -r TC_Estimate_qiangli

# Uncomment these lines if using the local /scratch filesystem
## Get results
#cp $SCRATCH/figure.png $SLURM_SUBMIT_DIR
#cp $SCRATCH/result.mat $SLURM_SUBMIT_DIR
## Remove scratch folder
#rm -rf $SCRATCH
~                                                                                                                                                                                                             
~                                                                                                                                                                                                             
~                   
