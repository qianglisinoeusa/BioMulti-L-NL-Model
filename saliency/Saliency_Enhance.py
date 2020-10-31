import os

import matlab.engine
names = matlab.engine.find_matlab()
eng = matlab.engine.connect_matlab()
eng.addpath(r'/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/saliency')

import cv2

img_path = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/MIT1003/SALIENCY_MAPS_QL'
save_path = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/MIT1003/SALIENCY_MAPS_QL/enhance/'    
for filename in os.listdir(img_path):
        image = cv2.imread(os.path.join(img_path, filename))
        s3 = eng.SaliencyEnhance(matlab.double(image.tolist()))
        while eng.isvalid(s3):
            pass
        cv2.imwrite(os.path.join(save_path, filename), Saliency_V) 
                  