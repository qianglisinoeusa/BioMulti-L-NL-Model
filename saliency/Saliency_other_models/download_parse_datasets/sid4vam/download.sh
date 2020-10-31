
wget --content-disposition --no-check-certificate "https://owncloud.cvc.uab.es/owncloud/index.php/s/wfK1SKIjoHEmtzK/download" -O dataset.zip
unzip dataset.zip -d .
cp -rf sid4vam_data/* ../../input
rm -rf dataset.zip
rm -rf sid4vam_data
