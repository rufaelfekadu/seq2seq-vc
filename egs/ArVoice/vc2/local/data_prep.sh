#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

fs=48000
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <spk> <data_dir>"
    echo "e.g.: $0 downloads/cms_us_slt_arctic slt data"
    echo ""
    echo "Options:"
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    exit 1
fi

set -euo pipefail

for set_name in ${train_set} ${dev_set} ${eval_set}; do
    
    [ ! -e "${data_dir}/${spk}_${set_name}" ] && mkdir -p "${data_dir}/${spk}_${set_name}"
    
    # set filenames
    scp="${data_dir}/${spk}_${set_name}/wav.scp"
    text="${data_dir}/${spk}_${set_name}/text"
    utt2spk="${data_dir}/${spk}_${set_name}/utt2spk"
    
    # check file existence
    [ -e "${scp}" ] && rm "${scp}"
    [ -e "${text}" ] && rm "${text}"
    [ -e "${utt2spk}" ] && rm "${utt2spk}"
    
    # Read from metadata.csv
    metadata="${db_root}/metadata.csv"
    
    # Check if metadata file exists
    if [ ! -f "${metadata}" ]; then
        echo "Error: Metadata file ${metadata} does not exist"
        exit 1
    fi
    
    # Skip header if present
    { tail -n +2 "${metadata}" 2>/dev/null || cat "${metadata}"; } | while IFS=, read -r path txt speaker; do
        # Remove quotes if present
        # path=$(echo "${path}" | sed 's/^"\(.*\)"$/\1/')
        # txt=$(echo "${txt}" | sed 's/^"\(.*\)"$/\1/')
        # speaker=$(echo "${speaker}" | sed 's/^"\(.*\)"$/\1/')
        # if speaker is different, skip
        if [ "${speaker}" != "${spk}" ]; then
            continue
        fi

        # if set_name not in path
        if [[ ! "${path}" == *"${set_name}"* ]]; then
            continue
        fi
        # Create ID from filename, split the basename using + as delimiter and take the second part
        id=$(basename "${path}" .wav)
        id=$(echo "${id}" | cut -d'+' -f2)


        # Check if file exists
        if [ -f "${db_root}/${path}" ]; then
            echo "${id} cat ${db_root}/${path} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
            echo "${id} ${txt}" >> "${text}"
            echo "${id} ${speaker}" >> "${utt2spk}"
        else
            echo "Warning: File ${db_root}/${path} not found"
        fi
    done
    
    echo "Successfully prepared  ${spk} ${set_name} data."
done

# Create the dev set if it doesn't exist
# Check if the dev set is empty (doesn't exist or has no wav.scp)
if [ ! -d "${data_dir}/${spk}_${dev_set}" ] || [ ! -s "${data_dir}/${spk}_${dev_set}/wav.scp" ]; then
    echo "Creating dev set from training data..."
    # Create temporary merged scp
    tmp_scp=$(mktemp)
    cat "${data_dir}/${spk}_${train_set}/wav.scp" > "${tmp_scp}"
    
    num_all=$(wc -l < "${tmp_scp}")
    num_dev=$((num_all / 10))
    num_train=$((num_all - num_dev))
    
    # Create directory for dev set
    mkdir -p "${data_dir}/${spk}_${dev_set}"
    
    utils/split_data.sh \
        --num_first "${num_train}" \
        --num_second "${num_dev}" \
        --shuffle "${shuffle}" \
        "${data_dir}/${spk}_${train_set}" \
        "${data_dir}/${spk}_${train_set}_temp" \
        "${data_dir}/${spk}_${dev_set}"
        
    rm "${tmp_scp}"

    # rename train set
    mv "${data_dir}/${spk}_${train_set}_temp" "${data_dir}/${spk}_${train_set}"
fi