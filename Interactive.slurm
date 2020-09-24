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
