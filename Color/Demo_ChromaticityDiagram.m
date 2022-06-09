clear all, close all, clc;
% 算出したXY値に該当するRGB値を曲線近似で求める場合
% Use Curve Fitting Toolbox to fit surface to sRGB data for vertices.
useCurvefit = false;
% 算出したXY値に該当するRGB値を近傍検索で求める場合
% Use Statistics and Machine Learning Toolbox to finds the nearest neighbor
% in reference color table for each query point in calculated XY value.
useStats = false;
% Both can be set to false, but execution time would be getting increased.
%% 画像データ読み込み
% Read image
img = imread('coloredchips.png');
%% 3D色空間、xy色度図表示
% Display 3-D color gamut and plot xy chromaticity diagram
myChromaticityDiagram(img, useCurvefit, useStats)
%% 画像データ読み込み
% Read image
img2 = imread('pears.png');
%% 3D色空間、xy色度図表示
% Display 3-D color gamut and plot xy chromaticity diagram
myChromaticityDiagram(img2, useCurvefit, useStats)
%   Copyright 2018 The MathWorks, Inc.