#Python
module load Anaconda3
conda create -n py3_env_updates python=3 numpy scipy matplotlib ipython jupyter jupyterlab scikit-learn 
#use later
module purge & module load Anaconda3
conda activate py3_env_updates
#conda install scikit-learn
conda install -c conda-forge hdf5storage

srun --pty -p LADON --mem=240000M -t 220:00:00 -c 46 bash
python TC_estimate.py 

#Python check virtual environment or list all estimated virtual environment

conda info --envs

#Matlab

module load MATLAB
salloc -n 56
srun --pty -x11 -p LADON -t 200:00:00 --mem 250000M  bash 

matlab -nodsiplay -nosplash -r 
