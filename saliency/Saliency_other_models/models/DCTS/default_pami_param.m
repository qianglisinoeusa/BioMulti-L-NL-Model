
function param = default_pami_param(sze)

param = {};

%%%%%%%%% scale parameters %%%%%%%%%%%%%%%

param.mapWidth = 64;               % this controls the size of the 'Center' scale
param.subtractMin = 1;           % 1 => (subtract min, divide by max) ; 0 => (just divide by max)

% signature
% param.type = 'dct';
% param.T = -1;



%Parametros de la gaussiana de suavizado(o autor varia o tamaño da
%gaussiana para obter distintos valores da ROC). Nos imos varias o tamaño
%da imaxe de entreda e a gaussiana faremola adaptativa para todos os
%algoritmos do mesmo xeito
param.maxhw = param.mapWidth -1;
param.sig = param.mapWidth * 0.04;


