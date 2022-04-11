#!/bin/bash

	## This code used for excuting pretrained denoised DNN for reconstructe BOLD signal. 
	
	
	## [Deep Universal Blind Image Denoising](https://www.micc.unifi.it/icpr2020/) <br> ICPR 2020


	## Environments
		#- Ubuntu 18.04
		#- [Tensorflow](http://www.tensorflow.org/) (>=1.8) 
		#- CUDA 11.2 & cuDNN
		#- Python 3.7

	## Author
	# UNIVERSITY OF VALENCIA
	# QIANGLI
	# Sep, 10, 2021, VALENCIA, SPAIN
	#

#function Denoised_DNN(InputPath, OutPath){
	#python main.py --sigma 35 --inputpath InputPath  --output OutPath
#}

function Denoised_DNN { args : InputPath , string lastName , OutPath } {
		InputPath = ${InputPath}
		OutPath = ${OutPath}
		python main.py --sigma 130 --inputpath InputPath  --output OutPath  
}

 
# Declare a string array with type
declare -a StringArray=("CSI1" "CSI2" "CSI3" "CSI4")
declare -a aStringArray=("LHPPA" "RHEarlyVis" "LHRSC"  "RHRSC"  "LHLOC"  "RHOPA"  "LHEarlyVis"  "LHOPA"  "RHPPA"  "RHLOC")
  	
# Read the array values with space
for subj in "${StringArray[@]}"; do
	#echo $subj
  	for region in "${aStringArray[@]}"; do
  		#echo $region
  		# Call DNN for Denoised
  		
  		InputPath="/home/qiangli/Python/Recon_Bold/Recon_Results/GaussianNB/$subj"_"$region"/"" 
  		#InputPath="/home/qiangli/Python/Recon_Bold/Recon_Results/LinearRegression/$subj"_"$region"/"" 
  		echo $InputPath
  		
	    OutPath="/home/qiangli/Python/Recon_Bold/GaussianNB/$subj"/"$region"/""
		#OutPath="/home/qiangli/Python/Recon_Bold/LinearRegression/$subj"/"$region"/""
		echo $OutPath

		if [ -d "$OutPath" ]; then
			echo "Directory already exists" ;
		else
			# attenation ` not ' or "
			`mkdir -p $OutPath`;
			echo "$OutPath directory is created"
		fi
		#conda install -c conda-forge tensorflow=1.14
  		python main.py --sigma 35 --inputpath $InputPath  --output $OutPath 
  		
	done
done

		            

#python main.py --sigma 35 --inputpath  /home/qiangli/Python/Recon_Bold/Recon_Results/BayesianRidge/CSI1_LHPPA/ --output /home/qiangli/Python/Recon_Bold/BayesianRidge/CSI1/LHPPA/
	