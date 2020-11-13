
wget --content-disposition --no-check-certificate "https://owncloud.cvc.uab.es/owncloud/index.php/s/zBMxbNSSUapoWWI/download" -O dataset.zip
unzip dataset.zip -d ..
rm dataset.zip
sh parse.sh
