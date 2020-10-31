    -------------------------------------------------------------------
---          Constrained Levy Exploration DEMO	    	 	       ---
---     Designed by Giuseppe Boccignone 				---
---     http://homes.di.unimi.it/~boccignone/GiuseppeBoccignone_webpage/Stochastic.html   ---
---	Dipartimento di Informatica 					---
---    		Universita' di Milano, Italy				---
---	    Copyright 2011, Universita' di Milano	 		---
---		   All rights reserved.					---
    -------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is hereby
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation.  For any
other uses of this software, in original or modified form, including
but not limited to distribution in whole or in part, specific prior
permission must be obtained from University of Salerno and the
authors.  These programs shall not be used, rewritten, or adapted as
the basis of a commercial software or hardware product without first
obtaining appropriate licenses from U. Salerno.  The University
makes no representations about the suitability of this software for
any purpose.  It is provided "AS IS" without express or implied
warranty.
---------------------------------------------------------------------------


===========GENERAL REMARKS========================================
The code is a simple Demo of a Constrained Levy Exploration (CLE)  scanpath generation experiment.

The experiment consists in putting into action a defined number of artificial observers, each generating a visual scanpath
(a sequence of fixations and saccades) on a given image using a slightly enhanced version of the CLE method described in Boccignone & Ferraro [1].
Enhancements concern;
                    - 1) Possibility of using more general alpha-stable
                       distributions [2] rather then stick to the Cauchy
                       distribution as in [1]
                    - 2) An informed strategy is employed to sample the
                       next gaze shift in that the choice of the next gaze location is
                       obtained through an internal simulation step: a number n of candidates gaze
                       shifts is preliminarly sampled and evaluated against a gain function [2]. 
                       The best among n candidate shift is eventually retained . 


===========SW INSTALLATION==========================================================
 
To  create the software library and run the demos:

1) unpack the compressed zip file in your working directory and cd to such directory (CLE)

	you will find the following directories:
        - /config:              Holds the configuration script all settings used in all parts of the code, enabling the exact
                                reproduction of the experiment at some future date
        - /doc: 		the reference papers
	- /img: 		color images to be processed
	- /results: 		to store segmentation results
        - /saltools:            the tools for computing saliency: for demo purposes
                                here you will find the Spectral Residual method
                                Store in this directory the methods you develop or download from external sources
        - /stats:               statistics tools
	- /visualization: 	some visualization tools
     	
2) add the path to this directory and subdirectories in your Matlab environment

3) edit if you like the /config/config_simple.m script for tuning the parameters of the experiment 
or just try it in the proposed configuration 

4) run demo program	
	>> runExperimentCLE
	

===============IMAGES=========================================
Some sample images are provided with the source code in the img directory 
   
===============DEMO PROGRAM======================================
The script

            runExperimentCLE

(1) sets the configuration script filename for setting the experiment
(2) sets the number of observers you want to simulate 
(2) generates a scanpath for each observer
===========
- cleGenerateScanpath(): 	
   Generates a visual scanpath by computing gaze shifts as Levy flights on
   any kind of saliency map (bottom-up or top-down) computed for the 
   given image. Basically a simple, but slightly enhanced, implementation of the algorithm
   described in the original paper of  Boccignone & Ferraro [1].
   The only variant with respect to [1] is the use of an internal
   simulation step along which a number of candidate gaze shifts is
   sampled [2].

   See the comments in each routine for details of what it does
   Settings for the experiment should be held in the configuration
   file.

- cleComputeSaliency():
   The function is a simple wrapper for salience computation. Executes some kind
   of salience computation algorithm which is defined from the parameter
   salType by calling the appropriate function. Here for simplicity only
   the Spectral Residual method has been considered. 
   If other methods need to be experimented, then you should extend the if...elseif...end
   control structure

- SpectralResidualSaliency():
  The function computes a salience map with the spectral residual method.
   The code is adapted from that published by X.Hou at 
   http://www.klab.caltech.edu/~xhou/projects/spectralResidual/spectralresidual.html
   The SR method provides comparable performance to other methods but at 
   a lower computational complexity end it is easy to code

- cleComputePotential():
   Computes the random walk potential as the function 
   $$ V(x,y)=\exp(-\tau _{V} s(x,y)) $$
   of a saliency map $s$

- cleShiftGazeLevy():
   Computes the gaze shift as a Levy walk step implemented trough a
   Langevin-like Stochastic Differential Equation (SDE)[1], where the random
   component is sampled from an alpha-stable distribution. The main
   difference from [1] is that the choice of the next gaze location is
   obtained through an internal simulation step: a number n of candidates gaze
   shifts is preliminarly sampled and evaluated against a gain function [2]. 
   The best among n candidate shift is eventually retained 

- cleWeightSal():
   
   weights the salience at a certain point (x,y) by using a Gaussian
   windows centered at that point

- stabrnd():
   Stable Random Number Generator. Based on the method of J.M. Chambers, C.L. Mallows and B.W.
   Stuck, "A Method for Simulating Stable Random Variables," 
   JASA 71 (1976): 340-4.

- sc(), label(): visualization functions

=====TIPS=================================================================

Different scanpath behaviors can be obtained by playing with parameters in the configuration script

=====REFERENCES=================================================================


[1] G. Boccignone and M. Ferraro, Modelling gaze shift as a constrained random walk, Physica A, vol. 331, no. 1, pp. 207-218, 2004.

 Bibtex:
 @article{boccignone2004modelling,
  title={Modelling gaze shift as a constrained random walk},
  author={Boccignone, G. and Ferraro, M.},
  journal={Physica A: Statistical Mechanics and its Applications},
  volume={331},
  number={1},
  pages={207--218},
  year={2004},
  publisher={Elsevier}
}

[] G. Boccignone and M. Ferraro, Feed and fly control of visual scanpaths for foveation image processing, Annals of telecommunications -  Annales des telecommunications  2012 (in press).

@article {boccignone2012annals,
   author = {Boccignone, Giuseppe and Ferraro, Mario},
   title = {Feed and fly control of visual scanpaths for foveation image processing},
   journal = {Annals of Telecommunications},
   publisher = {Springer Paris},
   pages = {1-17},
   url = {http://dx.doi.org/10.1007/s12243-012-0316-9},
}

==============================================================


Comments, suggestions, or questions should be sent to:
	Giuseppe.Boccignone@unimi.it
        http://homes.di.unimi.it/~boccignone/GiuseppeBoccignone_webpage/Stochastic.html

