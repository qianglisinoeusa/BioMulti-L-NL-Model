sudo apt-get install python python-dev python-opencv python-pip
sudo pip install -r requirements.txt
sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
echo 'edit scripts/constants.py and use your paths';
sudo nano scripts/constants.py;
wget https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/vgg16.pkl
wget https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/gen_modelWeights0090.npz
wget https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/discrim_modelWeights0090.npz


