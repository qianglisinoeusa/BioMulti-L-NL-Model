#How to set python3 or above version as default mode in the system(For me, Ubuntu18.04)
#Source: https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu


sudo update-alternatives --config python

#Will show you an error:

#update-alternatives: error: no alternatives for python3 

#You need to update your update-alternatives , then you will be able to 
#set your default python version.

sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2

#Then run:

sudo update-alternatives --config python

#Set python3.6 as default.

#Or use the following command to set python3.6 as default:

#sudo update-alternatives  --set python /usr/bin/python3.6