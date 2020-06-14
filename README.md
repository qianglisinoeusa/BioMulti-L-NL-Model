# 1. BioMulti-L-NL-Model

The biological possible multi-layer linear+nonlinear visual information processing model

DigitialBrain Version 3.0(Complex and Test Version)
Visual Computing

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
 

The Python version model inspired by matlab version. The matlab version also
get under request.


# 2. How to use it

@ caution: The code only can execute under conda envionment or virtual environment, otherwise,
it will cause errors.

a. First, run InstallDepedendent.sh to download and install dependent toolboxes.
b. Second, run main.sh to execute main funtion.


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