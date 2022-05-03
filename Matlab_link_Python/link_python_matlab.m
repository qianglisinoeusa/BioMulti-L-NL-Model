%% Matlab call Python function

%check matlab environment
if count(py.sys.path,'.') == 0
    insert(py.sys.path,int32(0),'.');
end

%clear classes
mod = py.importlib.import_module('cortc');
py.importlib.reload(mod);
T_corex=double(py.cortc.corex_tc_mat(x'))* log2(e);

%% Pythonfunction::cortc.py

%import os
%import sys
%sys.path.insert(0, './bio_corex')
%import corex as ce
%import warnings
%import time
%import default
%import linearcorex as lc
%import numpy as np

%def corex_tc_mat(x):
%    dim = x.shape[1]
%    layer1 = ce.Corex(n_hidden=8, dim_hidden=5,max_iter=60, n_repeat=10, eps=1e-5, verbose=True, n_cpu=7)
%    layer1.fit(x)
%    tc = layer1.tc
%    return tc

%def myfunc():
%    """Display message."""
%    return 'version 1'

%def corex_tc_lin(x):
%    out = lc.Corex(n_hidden=2, verbose=True)  
%    out.fit(np.linalg.pinv(x)) 
%    tc = out.tc
%    return tc  
    
