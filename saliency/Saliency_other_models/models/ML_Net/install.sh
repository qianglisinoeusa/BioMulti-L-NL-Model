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
mkdir tmp
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hMA3j5YY0zBkM0nhc6__7AxBDpqmNixV' -O tmp
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B3ZguV08iwjsOGFEWlRfZkVqaWs' -O tmp
mv tmp/vgg16_weights.h5 .
mv tmp/mlnet_salicon_weights.h5 .

