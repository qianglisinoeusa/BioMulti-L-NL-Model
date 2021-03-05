### BioMulti-L-NL-Model

<center>
<img width=600 height=200 src='imgs/mathmatics Neuroscience.jpg'><br/>
</center>

##The biological possible multi-layer linear+nonlinear visual information processing model

###Digitial Brain Visual Computing Version 2.0(Complex and Alpha Version)

<p>Human visual cortex inspired multi-layer LNL model. In this model, the main component are:</p>

<p>Nature Image --> VonKries Adaptation --> ATD  (Color processing phase)
Wavelets Transform --> Contrast sensivity function (CSF) --> Divisive
Normalization(DN)  --> Noise(Gaussian or Poisson)</p>

<p>Evalute and optimise model with TID2008 database -  one of image quality databases.</p>

<p>Redundancy redunction measure with Total Correlation(RBIG or Cortex module)</p>

<p>This model derivated two version script： Matlab, Python. In the future, I
want to implemented all of these code on C++ or Java. If our goal is 
simulate of primate brain, we need to implement all everything to High 
performance Computer(HPC) with big framework architecture(C/C++/Java).</p>
 

###Python Alpha Version

1. How to execute it

	@caution: The code only can execute under conda envionment or virtual environment, otherwise,
	it will cause errors. Pytorch only works under the virtual environment.<br/>

	option 1: <br/>

	*source ~/.bashrc* : (base) this conda environment.<br/>
	*conda deactivate* : quit conda environment. <br/>


	option 2:<br/>

	*cd virutal environment.* <br/>
	*python3 -m venv pytorch-BioMulti-L-NL* <br/>  
	*source pytorch-BioMulti-L-NL/bin/activate* <br/>
	*deactivate* <br/>


	Run <b>InstallDepedendent.sh</b>to download and install dependent toolboxes.<br/>
	Run <b>main.sh</b> to execute main funtion.<br/>

### Python Beta Version

The beta version running environment same with alpha version.<br/>

Run <b>main.sh</b> under bash environment.<br/>

### Python Beta Version L+NL model parameters optimization

The model parameters optimization can be done with Jacobian respect each parameters. The main function that implemented with <b>jacobian.py</b> and <b>optimization.py</b>.  The demo of how to optimization parameters in the model, please check here:

https://github.com/matthias-k/pysaliency/tree/master/pysaliency.


### Requierment toolboxes(see requirements.txt):

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


### Matlab Alpha Version

1. Dependent toolboxes

	[*matlabPyrtools*](https://github.com/LabForComputationalVision/matlabPyrTools)<br/>
	*ColorE*<br/>
	*Hdrvdp*<br/>
	*BioMulti-L-NL-Model*<br/>
	[*TID2008 database*](http://www.ponomarenko.info/tid2008.htm)<br/>

2. How to run it

	*TID2008.m*: evaluate LNL model with TID2008 dataset.<br/>
	The main function will call *RLV.m* and *simple_model_rlv.m* function from the path then plot the results. <br/>

###The parameters still can optimize in the future.

### If you think this project can help you or you can use something from this project then please consider cite below related paper:


```
@article{Alex20,
author = {Gomez-Villa, Alex and Bertalmío, Marcelo and Malo, Jesús},
year = {2020},
month = {03},
pages = {},
title = {Visual Information flow in Wilson-Cowan networks},
volume = {123},
journal = {Journal of Neurophysiology},
doi = {10.1152/jn.00487.2019}
}
```


```
@article{Marina17,
author = {Martinez-Garcia, Marina and Cyriac, Praveen and Batard, Thomas and Bertalmío, Marcelo and Malo, Jesús},
year = {2017},
month = {11},
pages = {},
title = {Derivatives and Inverse of Cascaded Linear+Nonlinear Neural Models},
volume = {13},
journal = {PLoS ONE},
doi = {10.1371/journal.pone.0201326}
}
```


```
@InProceedings{Qiang20,
author="Li, Qiang and Malo, Jesus",
title="Canonical Retina-to-Cortex Vision Model Ready for Automatic Differentiation",
booktitle="Brain Informatics",
year="2020",
publisher="Springer International Publishing",
pages="329--337",
isbn="978-3-030-59277-6"
}
```
 

### If you have any question, please contact me.

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
