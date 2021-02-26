#How to link  R-3.4.4 to R-4.0.2


# Check location of old version R
which R 

# Install latest version R

# Link old version R to new version R

alias R = /home/liqiang/Downloads/R-4.0.2/bin/R

#After adding the above in the file, run source ~/.bashrc or source ~/.bash_aliases.
source ~/.bashrc


# If you want to back R-3, you can use

unalias R


python run_model.py --input_path=/home/liqiang/Downloads/blurred_and_noisy_images/DnCNN-tensorflow/data/denoised   --output_path=/home/liqiang/Downloads/blurred_and_noisy_images/FIN_RESULT/