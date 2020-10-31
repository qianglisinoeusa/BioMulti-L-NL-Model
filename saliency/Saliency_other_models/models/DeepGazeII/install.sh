sudo apt-get install python3 python3-dev python3-opencv python3-pip
sudo pip install -r requirements.txt
mkdir tmp
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kYUwoatqQUS5EabeeSDc6gRmCysnVZ6N' -O tmp
unzip deep_gaze.zip -d tmp
mv tmp/*.ckpt* .

