#!/bin/bash
# download raw cityscapes data leftimg8bit

# parse argument
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -u|--username)
    USERNAME="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--password)
    PASSWORD="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# get cookies
wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=${USERNAME}&password=${PASSWORD}&submit=Login" https://www.cityscapes-dataset.com/login/

# download data
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

# unzip
unzip leftImg8bit_trainvaltest.zip

# rename
mv leftImg8bit_trainvaltest leftImg8bit

# remove intermediate files
rm index.html cookies.txt leftImg8bit_trainvaltest.zip
