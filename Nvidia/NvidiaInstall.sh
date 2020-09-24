#Step1: If you have GPU support hardware. Firstly go to
#software and updates center, then go to additional drivers select
#you right Nivdia-drivers.

#Step2: Install NVIDIA Graphics Driver via apt-get(cuda-9.0)

sudo apt-get install nvidia-384 nvidia-modprobe

#Reboot your machine but enter BIOS to disable Secure Boot. 

#Step3: Install CUDA 9.0 via Runfile

wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run

chmod +x cuda_9.0.176_384.81_linux-run
./cuda_9.0.176_384.81_linux-run --extract=$HOME
sudo ./cuda-linux.9.0.176-22781540.run
sudo ./cuda-samples.9.0.176-22781540-linux.run

#After the installation finishes, configure the runtime library.

sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
sudo ldconfig

sudo vim /etc/environment

#and then add :/usr/local/cuda/bin (including the ":") at the end of the PATH="/blah:/blah/blah" string (inside the quotes).

#Step4: Install cuDNN 7.0

# Go to the cuDNN download page (need registration) and select the latest cuDNN 7.0.* version made for CUDA 9.0.

# Download all 3 .deb files: the runtime library, the developer library, and the code samples library for Ubuntu 16.04.

# In your download folder, install them in the same order:

sudo dpkg -i libcudnn7_7.0.5.15–1+cuda9.0_amd64.deb (the runtime library),

sudo dpkg -i libcudnn7-dev_7.0.5.15–1+cuda9.0_amd64.deb (the developer library), and

sudo dpkg -i libcudnn7-doc_7.0.5.15–1+cuda9.0_amd64.deb (the code samples).

# Step5: Configure the CUDA and cuDNN library paths


# Sources:https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e
