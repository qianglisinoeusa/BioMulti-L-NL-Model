sudo apt-get install python python-dev python-opencv python-pip
sudo pip install --upgrade Theano==0.9.0
sudo pip install --upgrade Keras==1.1.0
sudo pip install h5py

echo "
{

    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano",
    "image_dim_ordering": "th"
}" > $HOME/.keras/keras.json;
mkdir weights
wget https://github.com/marcellacornia/sam/releases/download/1.0/sam-vgg_salicon_weights.pkl -P weights
wget https://github.com/marcellacornia/sam/releases/download/1.0/sam-resnet_salicon_weights.pkl -P weights
wget https://github.com/marcellacornia/sam/releases/download/1.0/sam-resnet_salicon2017_weights.pkl -P weights
