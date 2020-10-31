------------------------------------------------------------------------------

Matlab tools for "Low-level spatiochromatic grouping for saliency estimation"

Contact: Naila Murray at <naila.murray@xrce.xerox.com>
 
------------------------------------------------------------------------------




--------------


Contents

--------------


This code package includes the following files:


- SIM_demo.m: loads a sample image and returns and displays a saliency map.


- SIM.m: converts the image to the opponent colour space, generates a saliency map for each channel and combines these maps to produce the final saliency map.


- rgb2opponent.m: converts the image to the opponent colour space.


- generate_csf.m: returns the value of the csf at a specific center-surround contrast energy and spatial scale.


- GT.m: performs the forward DWT and GT on one channel of an image and applies the ECSF.

- blockMatching.<c,mexa64,mexglx,mexw64>: block matching source code and executables. Determines association field.

- norm_center_contrast.m: computes normalized center contrast
- IDWT.m: performs the inverse DWT on one channel of an image.


- symmetric_filtering.m: performs 1D Gabor filtering with symmetric edge handling.
- mirroring.m: helper function to symmetric_filtering.m
- add_padding.m: pads image to ensure dimensions are powers of 2.

- 3.jpg and 35.jpg: sample images (from dataset of Bruce et al., Saliency Based on Information Maximization. Advances in Neural Information Processing Systems 18, 2006).




---------------

Getting Started

---------------


To run the demo, execute SIM_demo.m



