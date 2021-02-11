import os
import numpy as np
import cv2
import glob

def preprocess_imgs(dim, img_Path, save_Path):
    for filename in os.listdir(img_Path):
        name, extension = os.path.splitext(filename)
        or_image = cv2.imread(os.path.join(img_Path, filename))
        image= cv2.resize(or_image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(save_Path, name + '.png'), image) 
        
if __name__ == '__main__':
    
    datasets = ['MIT1003', 'MIT300',  'TORONTO']
    dim = (682, 682)
    for dataset in datasets:
                
                print(dataset)

                MIT1003_dataset = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/mit1003/images/'
                copy_to_path1 = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/mit1003/images/'
                preprocess_imgs(dim, MIT1003_dataset, copy_to_path1)
                #os.remove(glob.glob(os.path.join(MIT1003_dataset, '*.jpeg')))
            
                MIT300_dataset = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/mit300/images/'
                copy_to_path2 = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/mit300/images/'
                preprocess_imgs(dim, MIT300_dataset,copy_to_path2)
                #os.remove(glob.glob(os.path.join(MIT300_dataset, '*.jpg')))
            
                TORONTO_dataset = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/toronto/images/'
                copy_to_path3 = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/toronto/images/'
                preprocess_imgs(dim, TORONTO_dataset,copy_to_path3)
                #os.remove(glob.glob(os.path.join(TORONTO_dataset, '*.jpg')))
            
    print('DONE')
        