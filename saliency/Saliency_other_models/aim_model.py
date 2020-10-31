import pysaliency

dataset_location = 'datasets'
model_location = 'models'

path2 = '/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/saliency/Saliency_other_models/AIM_model/AIM_Result'
    
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=dataset_location)
aim = pysaliency.AIM(location=model_location)
saliency_map = aim.saliency_map(mit_stimuli.stimuli[0])

cv2.imwrite(os.path.join(path2 ,filename), saliency_map*255) 
        