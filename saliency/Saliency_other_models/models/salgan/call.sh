cd scripts; 
sudo THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,lib.cnmem=1 /usr/bin/python 03-predict.py; 
cd ..;
