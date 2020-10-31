% Build the hard-coded 48x64 Quaternion DCT saliency .mex-interface
% (if you want faster code, then you should add further optimization
%  parameters to the mex compiler)
%
% @author  B. Schauerte
% @date:   2011
% @url:    http://cvhci.anthropomatik.kit.edu/~bschauer/

% compile the hard-coded DCT type-II and type-III
mex -c dct_type2_48.cpp
mex -c dct_type2_64.cpp
mex -c dct_type3_48.cpp
mex -c dct_type3_64.cpp

% compile the .mex-files/interfaces
mex -D__MEX dct_48_64.cpp dct_type2_48.obj dct_type2_64.obj dct_type3_48.obj dct_type3_64.obj
mex -D__MEX hamilton_product.cpp 
mex -D__MEX signum.cpp 
mex -c dct_48_64.cpp
mex -D__MEX qdct_saliency_48_64_nofilter.cpp dct_type2_48.obj dct_type2_64.obj dct_type3_48.obj dct_type3_64.obj dct_48_64.obj -output qdct_saliency_48_64
