#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

fs=24000
num_dev=5
num_eval=5
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
    echo "Usage: $0 [Options] <db_root> <spk> <data_dir>"
    echo ""
    echo "Options:"
    echo "    --fs: target sampling rate (default=22050)."
    echo "    --num_dev: number of development uttreances (default=10)."
    echo "    --num_eval: number of evaluation uttreances (default=10)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffule to make dev & eval set (default=true)."
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e "${db_root}/${spk}" ] && \
    echo "${spk} does not exist." >&2 && exit 1;

[ ! -e "${data_dir}/all_${spk}" ] && mkdir -p "${data_dir}/all_${spk}"
[ ! -e "${data_dir}/test_${spk}" ] && mkdir -p "${data_dir}/test_${spk}"
[ ! -e "${data_dir}/non_test_${spk}" ] && mkdir -p "${data_dir}/non_test_${spk}"

# set filenames
scp="${data_dir}/all_${spk}/wav.scp"
test_scp="${data_dir}/test_${spk}/wav.scp"
non_test_scp="${data_dir}/non_test_${spk}/wav.scp"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${test_scp}" ] && rm "${test_scp}"
[ -e "${non_test_scp}" ] && rm "${non_test_scp}"

# make all scp
find "${db_root}/${spk}" -follow -name "*.wav" | sort | while read -r filename; do
    id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
    
    # Separate test files and non-test files
    if [[ "${filename}" == *"test"* ]]; then
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${test_scp}"
    else
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${non_test_scp}"
    fi
done

# Get the number of test files
num_test=$(wc -l < "${test_scp}")
echo "Found ${num_test} test files."

# If we have enough test files, use them directly
if [ ${num_test} -ge ${num_eval} ]; then
    # Split non-test files into train and dev
    num_non_test=$(wc -l < "${non_test_scp}")
    num_train=$((num_non_test - num_dev))
    
    # Split non-test files into train and dev
    utils/split_data.sh \
        --num_first "${num_train}" \
        --num_second "${num_dev}" \
        --shuffle "${shuffle}" \
        "${data_dir}/non_test_${spk}" \
        "${data_dir}/${train_set}" \
        "${data_dir}/${dev_set}"
    
    # Use test files for eval
    if [ ${num_test} -gt ${num_eval} ]; then
        # If we have more test files than needed, take a subset
        mkdir -p "${data_dir}/${eval_set}"
        head -n ${num_eval} "${test_scp}" > "${data_dir}/${eval_set}/wav.scp"
    else
        # If we have exactly the right number, just copy
        mkdir -p "${data_dir}/${eval_set}"
        mv "$test_scp" "${data_dir}/${eval_set}/"
    fi
else
    # Not enough test files, fall back to original split method
    echo "Warning: Not enough files with 'test' in their name (found ${num_test}, needed ${num_eval})."
    echo "Falling back to random splitting."
    
    num_all=$(wc -l < "${scp}")
    num_deveval=$((num_dev + num_eval))
    num_train=$((num_all - num_deveval))
    
    if [ ${num_eval} -ne 0 ]; then
        utils/split_data.sh \
            --num_first "${num_train}" \
            --num_second "${num_deveval}" \
            --shuffle "${shuffle}" \
            "${data_dir}/all_${spk}" \
            "${data_dir}/${train_set}" \
            "${data_dir}/deveval_${spk}"
        utils/split_data.sh \
            --num_first "${num_dev}" \
            --num_second "${num_eval}" \
            --shuffle "${shuffle}" \
            "${data_dir}/deveval_${spk}" \
            "${data_dir}/${dev_set}" \
            "${data_dir}/${eval_set}"
    else
        utils/split_data.sh \
            --num_first "${num_train}" \
            --num_second "${num_deveval}" \
            --shuffle "${shuffle}" \
            "${data_dir}/all_${spk}" \
            "${data_dir}/${train_set}" \
            "${data_dir}/${dev_set}"
        cp -r "${data_dir}/${dev_set}" "${data_dir}/${eval_set}"
    fi
fi

# remove tmp directories
rm -rf "${data_dir}/all_${spk}"
rm -rf "${data_dir}/deveval_${spk}"
rm -rf "${data_dir}/test_${spk}"
rm -rf "${data_dir}/non_test_${spk}"

echo "Successfully prepared data."