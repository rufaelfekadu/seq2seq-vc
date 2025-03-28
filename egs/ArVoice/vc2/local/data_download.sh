#!/usr/bin/env bash
set -e

# Copyright 2023 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <db_root_dir>"
    exit 1
fi

# download dataset
cwd=`pwd`
if [ ! -e ${db}/ArVoice.done ]; then
    mkdir -p ${db}
    cd ${db}
    # make a symbolic link to the dataset folder
    cp -r /home/hawau/projects/ArVoice ArVoice
    touch ArVoice.done
else
    echo "Already exists. Skip download."
fi
