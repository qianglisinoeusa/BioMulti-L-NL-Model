
wget --content-disposition --no-check-certificate "https://owncloud.cvc.uab.es/owncloud/index.php/s/CTcpBpTbL7JfEu3/download" -O dataset.zip
unzip dataset.zip -d ..
rm dataset.zip
sh parse.sh
