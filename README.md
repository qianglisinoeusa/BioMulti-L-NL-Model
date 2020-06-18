### BioMulti-L-NL-Model

The biological possible multi-layer linear+nonlinear visual information processing model

Digitial Brain Visual Computing Version 2.0(Complex and Alpha Version)

Human visual inspired multi-layer LNL model. In this model, the main 
component are:

Nature Image --> VonKries Adaptation --> ATD  (Color processing phase)
Wavelets Transform --> Contrast sensivity function (CSF) --> Divisive
Normalization(DN)  --> Noise(Gaussian or Poisson)

Evalute and optimise model with TID2008 database -  one of image quality databases.

Redundancy redunction measure with Total Correlation(RBIG or Cortex module) - under constructation

This model derivated two version script： Matlab, Python. In the future, I
want to implemented all of these code on C++ or Java. If our goal is 
simulate of primate brain, we need to implement all everything to High 
performance Computer(HPC) with big framework architecture(C/C++/Java).
 

#### Python Alpha Version



1. How to execute it

@caution: The code only can execute under conda envionment or virtual environment, otherwise,
it will cause errors. Pytorch only works under the virtual environment.<br/>

option 1: <br/>

source ~/.bashrc : (base) this conda environment.<br/>
conda deactivate : quit conda environment. <br/>


option 2:<br/>

cd virutal environment. <br/>
python3 -m venv pytorch-BioMulti-L-NL <br/>  
source pytorch-BioMulti-L-NL/bin/activate <br/>
deactivate <br/>


Run InstallDepedendent.sh to download and install dependent toolboxes.<br/>
Run main.sh to execute main funtion.<br/>

#### Python Beta Version

The beta version running environment same with alpha version.<br/>

Run main.sh under bash environment.<br/>



2. Requierment toolboxes(see requirements.txt):<br/>

numpy<br/>
NeuroTools<br/>
statsmodels<br/>
pyrtools<br/>
MotionClouds<br/>
tensorflow<br/>
pytorch<br/>
PyWavelets<br/>
colour-science<br/>
scipy<br/>
opencv<br/>
SLIP<br/>
PyTorchSteerablePyramid<br/>
PIL<br/>
tqdm<br/>
LogGabor<br/>
nt_toolbox<br/>


#### Matlab Alpha Version

1. Dependent toolboxes

[matlabPyrtools](https://github.com/LabForComputationalVision/matlabPyrTools)<br/>
ColorE<br/>
Hdrvdp<br/>
BioMulti-L-NL-Model<br/>
[TID2008 database](http://www.ponomarenko.info/tid2008.htm)<br/>

2. How to run it

TID2008.m: evaluate LNL model with TID2008 dataset.<br/>
The main function will call RLV.m and simple_model_rlv.m function from the path then plot the results. <br/>

The parameters still can optimize in the future.

----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is here
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.

@Copyright(c) QiangLi, 2020, Valencia, Spain.