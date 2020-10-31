# Run metrics code #

This is the Matlab saliency code able to process saliency maps and metrics. 

Run metrics from "input/", Example for test dataset:  
    
    main('test',0);

This will export a csv in "output/" (main.m automatically runs see_results.m after storing results).
    
Images, fixation maps and saliency maps should be saved in the following directories (by default):<br/>
    
    input/images/DATASET_NAME
    input/bmaps/DATASET_NAME
    input/dmaps/DATASET_NAME
    input/smaps/DATASET_NAME/MODEL_NAME
    
Although only necessary folders for computing saliency metrics are "images", "bmaps", "dmaps" and "smaps". Other metrics require region binary masks "mmaps" or scanpaths ("input/smaps/DATASET_NAME/scanpaths" for GT, "input/smaps/DATASET_NAME/MODEL_NAME/scanpaths" for model scanpaths).<br/>
To run any dataset, copy your dataset files and make sure you have the same folder structure as in "test".<br/>
   
# Run Saliency models #

Run saliency maps from "models/", Example for test dataset: 

    get_smaps('models',{'test'});
    
Before running saliency models, first try to delete them, e.g. "input/smaps/test/SIM"<br/>
Note: To add another model, make sure each model runs a matlab file with same format (input and output of function) and prefix "saliency_MODEL_NAME" as in "saliency_sim.m". You can also run shell commands through matlab for python-based models. We have included an example code of all saliency models to make your tests (try to run install.sh in every subfolder before running them).

# Download Datasets #

Use the shell commands (download_parse_datasets/DATASET_NAME/download.sh) for downloading and parsing eye-tracking data (images, fixation maps and scanpaths) of previous experiments of Toronto (Bruce & Tsotsos, 2006), MIT1003 (Judd et al., 2009), KTH (Kootstra et al., 2011), CAT2000 Pattern (Borji & Itti, 2015) and SID4VAM (Berga et al., 2019). You can download the saliency maps from [my owncloud](https://owncloud.cvc.uab.es/owncloud/index.php/s/IJLBgMtcBvzH4vU).
    
# This is a fork of official MIT metrics code #

Code from https://github.com/cvzoya/saliency is found in "include/saliency-master/"

Various code related to the MIT saliency benchmark website http://saliency.mit.edu will be found there. 
Please contact saliency@mit.edu with any questions.
If you use any of this code, please cite: 
<pre>
@article{salMetrics_Bylinskii,
    title    = {What do different evaluation metrics tell us about saliency models?},
    author   = {Zoya Bylinskii and Tilke Judd and Aude Oliva and Antonio Torralba and Fr{\'e}do Durand},
    journal  = {arXiv preprint arXiv:1604.03605},
    year     = {2016}
}
</pre>
