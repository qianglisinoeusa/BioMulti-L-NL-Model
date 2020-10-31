==============================================================================
Matlab tools for saccadic model of visual attention
  
Copyright (c) 2015 University of Rennes 1 / Olivier Le Meur
==============================================================================

==============================================================================
// Demo 
==============================================================================
Sample code can be found in the script the following function:
function scriptForGeneratingScanpaths(directoryOrigPict, directoryOrigSal, directoryOutput, nbScanpaths, nbFixations, timeToRecover, nbCandidates, nppd, sceneType)

* directoryOrigPict: input pictures (in jpg format)
* directoryOrigSal: input saliency (in pgm format) with the same filename as original picture
* directoryOutput: output results (scanptah-based saliency maps + file containing the scanpaths
* nbScanpaths: number of scanpaths to compute
* nbFixations: number of fixations per scanpath
* nbCandidates: number of candidates to draw from the distribution. nbCandidates=1, highest dispersion between observers, highest stochasticity; nbCandidates >> 1, the model becomes almost deterministic (corresponds to the Bayesian searcher).
* nppd: number of pixels per degree.
* timeToRecover: 8 fixations. -1 => WTA there is no recovery
* sceneType could be: 'naturalScenes', 'webPages', 'faces', 'landscapes'

This code requires a RELEVANT input saliency map. We suggest to use the average of the GBVS and RARE2012 saliency maps. See papers for details.

==============================================================================
// How to run the soft
==============================================================================
scriptForGeneratingScanpaths('./imgSource/', './imgSaliency/', './results/', 10, 10, 8, 5, 22, 'naturalScenes')


==============================================================================
// Bibtex reference and web page
==============================================================================
The source code corresponds to the following papers. Please cite both papers if you intend to use this software.

Le Meur, O., & Liu, Z. (2015). Saccadic model of eye movements for free-viewing condition. Vision research, 116, 152-164.

@article{LeMeur2015,
  title={Saccadic model of eye movements for free-viewing condition},
  author={Le Meur, Olivier and Liu, Zhi},
  journal={Vision research},
  volume={116},
  pages={152--164},
  year={2015},
  publisher={Elsevier}
}

Le Meur, O., & Coutrot, A. (2016). Introducing context-dependent and spatially-variant viewing biases in saccadic models. Vision research, 121, 72-84.

@article{LeMeur2016,
  title={Introducing context-dependent and spatially-variant viewing biases in saccadic models},
  author={Le Meur, Olivier and Coutrot, Antoine},
  journal={Vision research},
  volume={121},
  pages={72--84},
  year={2016},
  publisher={Elsevier}
}


==============================================================================
// Contact:
==============================================================================
Please contact Olivier Le Meur (olemeur@irisa.fr) in case you have any questions or comments.

This code and data is free to use subject to the disclaimers contained within for academic and non-profit research purposes. 

