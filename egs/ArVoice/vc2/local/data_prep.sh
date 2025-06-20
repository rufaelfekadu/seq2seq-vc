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
tgtspk="ar-XA-Wavenet-A" 

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
spk_found=false
for subgroup in "${db_root}"/*/; do
    if [ -d "${subgroup}${spk}" ]; then
        spk_found=true
        break
    fi
done
[ "${spk_found}" = false ] && \
    echo "${spk} does not exist in any subgroup." >&2 && exit 1;

[ ! -e "${data_dir}/all_${spk}" ] && mkdir -p "${data_dir}/all_${spk}"
[ ! -e "${data_dir}/test_${spk}" ] && mkdir -p "${data_dir}/test_${spk}"
[ ! -e "${data_dir}/non_test_${spk}" ] && mkdir -p "${data_dir}/non_test_${spk}"

# do the same for the tgt speaker
[ ! -e "${data_dir}/all_${tgtspk}" ] && mkdir -p "${data_dir}/all_${tgtspk}"
[ ! -e "${data_dir}/test_${tgtspk}" ] && mkdir -p "${data_dir}/test_${tgtspk}"
[ ! -e "${data_dir}/non_test_${tgtspk}" ]&& mkdir -p "${data_dir}/non_test_${tgtspk}"


# set filenames
scp="${data_dir}/all_${spk}/wav.scp"
test_scp="${data_dir}/test_${spk}/wav.scp"
non_test_scp="${data_dir}/non_test_${spk}/wav.scp"

tgt_scp="${data_dir}/all_${tgtspk}/wav.scp"
tgt_test_scp="${data_dir}/test_${tgtspk}/wav.scp"
tgt_non_test_scp="${data_dir}/non_test_${tgtspk}/wav.scp"


# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${test_scp}" ] && rm "${test_scp}"
[ -e "${non_test_scp}" ] && rm "${non_test_scp}"

# make all scp
# for subgroup in "${db_root}"/ArVoice_syn-${spk}/; do
#     if [ -d "${subgroup}${spk}" ]; then

find "${db_root}"/Arvoice_syn-"${spk}"/"${spk}" -follow -name "*.wav" | sort | while read -r filename; do
    id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
    
    # Separate test files and non-test files
    if [[ "${filename}" == *"test"* ]]; then
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${test_scp}"
    else
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${non_test_scp}"
    fi
done

find "${db_root}"/Arvoice_syn-"${spk}"/"${tgt_spk}" -follow -name "*.wav" | sort | while read -r filename; do
    id=$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")
    echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${scp}"
    
    # Separate test files and non-test files
    if [[ "${filename}" == *"test"* ]]; then
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${test_scp}"
    else
        echo "${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |" >> "${non_test_scp}"
    fi
done
#     fi
# done

# Get the number of test files
num_test=$(wc -l < "${test_scp}")
echo "Found ${num_test} test files."

# if no non-test files exist just prepare test set
if [ ! -s "${non_test_scp}" ]; then
    echo "No non-test files found. Using all files as test set."
    rm -rf "${data_dir}/all_${spk}"
    rm -rf "${data_dir}/non_test_${spk}"
    exit 0
fi


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
    # If we have exactly the right number or less, just copy
    echo "Using all ${num_test} test files for evaluation."
    mkdir -p "${data_dir}/${eval_set}"
    mv "$test_scp" "${data_dir}/${eval_set}/"
fi

# remove tmp directories
rm -rf "${data_dir}/all_${spk}"
rm -rf "${data_dir}/deveval_${spk}"
rm -rf "${data_dir}/test_${spk}"
rm -rf "${data_dir}/non_test_${spk}"

echo "Successfully prepared data."