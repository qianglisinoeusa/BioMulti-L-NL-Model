1. GENERAL

  Not much to say at the moment. If you are interested in the topic, then
  please visit my website (http://cvhci.anthropomatik.kit.edu/~bschauer/).

  If you use any of this work in scientific research or as part of a larger
  software system, you are requested to cite the use in any related 
  publications or technical documentation. The work is based upon:

  [1] B. Schauerte, and R. Stiefelhagen, "Quaternion DCT Spectral Saliency: 
      Predicting Human Gaze using Quaternion DCT Image Signatures and Face
      Detection," in IEEE Workshop on Applications of Computer Vision (WACV),
      2012.

2. INSTALLATION

2.1. QUATERNION METHODS AND THE QTFM

  If you want to use the Quaternion spectral saliency methods, then you need
  the "Quaternion Toolbox for Matlab" (QTFM). Most importantly, for the 
  .m-file implementation of the quaternion dct spectral saliency you need to
  patch the implementation by adding the .m-files in qtfm/@quaternion to the
  corresponding folder of the QTFM.

2.2. OPTIMIZED C/C++ IMPLEMENTATION (.MEX)

  The necessary .mex files for the optimized implementations can be generated
  by running build.m. However, this should also be done automatically when
  calling spectral_saliency_multichannel, if needed.

3. AUTHORS AND CONTACT INFORMATION

  B. Schauerte <boris.schauerte@kit.edu> 
