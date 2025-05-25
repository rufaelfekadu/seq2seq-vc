#!/usr/bin/env bash

# Copyright 2023 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=8      # number of parallel jobs in feature extraction

conf=conf/aas_vc.melmelmel.v1.yaml

# dataset configuration
db_root=/workspace/ArVoice-syn
dumpdir=dump                # directory to dump full features
srcspk=ar-XA-Wavenet-C                  # available speakers: "clb" "bdl"
trgspk=ar-XA-Wavenet-D                  # available speakers: "slt" "rms"
stats_ext=h5
norm_name=self                  # used to specify normalized data.
                            # Ex: `judy` for normalization with pretrained model, `self` for self-normalization

src_feat=mel
trg_feat=mel
dp_feat=mel

train_duration_dir=none     # need to be properly set if FS2-VC is used
dev_duration_dir=none       # need to be properly set if FS2-VC is used

# pretrained model related
pretrained_model_checkpoint=

# training related setting
tag=""     # tag for directory to save model
resume="/workspace/seq2seq-vc/egs/ArVoice/vc2/exp/ar-XA-Wavenet-B_ar-XA-Wavenet-C_male_male/checkpoint-10632steps.pkl"  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train" # name of training data directory
dev_set="dev"           # name of development data directory
eval_set="test"         # name of evaluation data directory
shuffle=false
num_dev=50
num_eval=50
set -euo pipefail

# sanity check for norm_name and pretrained_model_checkpoint
if [ -z ${norm_name} ]; then
    echo "Please specify --norm_name ."
    exit 1
elif [ ${norm_name} == "self" ]; then
    if [ ! -z ${pretrained_model_checkpoint} ]; then
        echo "You cannot specify pretrained_model_checkpoint and norm_name=self simultaneously."
        exit 1
    fi
    src_stats="${dumpdir}/${srcspk}_train/stats.${stats_ext}"
    trg_stats="${dumpdir}/${trgspk}_train/stats.${stats_ext}"
else
    if [ -z ${pretrained_model_checkpoint} ]; then
        echo "Please specify the pretrained model checkpoint."
        exit 1
    fi
    pretrained_model_dir="$(dirname ${pretrained_model_checkpoint})"
    src_stats="${pretrained_model_dir}/stats.${stats_ext}"
    trg_stats="${pretrained_model_dir}/stats.${stats_ext}"
fi

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#     echo "stage -1: Data & Pretrained Model Download"

#     # download dataset
#     # local/data_download.sh "downloads"

#     # download pretrained vocoder
#     utils/hf_download.py --repo_id "unilight/hificaptain-vc" --outdir "downloads" --filename "pwg_jp_female/checkpoint-400000steps.pkl"
#     utils/hf_download.py --repo_id "unilight/hificaptain-vc" --outdir "downloads" --filename "pwg_jp_female/config.yml"
#     utils/hf_download.py --repo_id "unilight/hificaptain-vc" --outdir "downloads" --filename "pwg_jp_female/stats.h5"

#     # download pretrained aas model
#     utils/hf_download.py --repo_id "unilight/hificaptain-vc" --outdir "exp" --filename "male_female_aas_vc_mel_pretrained/checkpoint-50000steps.pkl"
#     utils/hf_download.py --repo_id "unilight/hificaptain-vc" --outdir "exp" --filename "male_female_aas_vc_mel_pretrained/config.yml"
#     utils/hf_download.py --repo_id "unilight/hificaptain-vc" --outdir "exp" --filename "male_female_aas_vc_mel_pretrained/stats.h5"
# fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    for spk in ${srcspk} ${trgspk}; do
        local/data_prep.sh \
            --fs "$(yq ".sampling_rate" "${conf}")" \
            --shuffle "${shuffle}" \
            --num_dev "${num_dev}" \
            --num_eval "${num_eval}" \
            --train_set "${spk}_${train_set}" \
            --dev_set "${spk}_${dev_set}" \
            --eval_set "${spk}_${eval_set}" \
            "${db_root}" "${spk}" data
    done
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Feature extraction"

    # if norm_name=self, then use $conf; else use config in pretrained_model_dir
    if [ ${norm_name} == "self" ]; then
        config_for_feature_extraction="${conf}"
    else
        config_for_feature_extraction="${pretrained_model_dir}/config.yml"
    fi

    # extract raw features
    pids=()
    for name in "${srcspk}_${train_set}" "${srcspk}_${dev_set}" "${srcspk}_${eval_set}" "${trgspk}_${train_set}" "${trgspk}_${dev_set}" "${trgspk}_${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/raw" ] && mkdir -p "${dumpdir}/${name}/raw"
        echo "Feature extraction start. See the progress via ${dumpdir}/${name}/raw/preprocessing.*.log."
        utils/make_subset_data.sh "data/${name}" "${n_jobs}" "${dumpdir}/${name}/raw"
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/raw/preprocessing.JOB.log" \
            preprocess.py \
                --config "${config_for_feature_extraction}" \
                --scp "${dumpdir}/${name}/raw/wav.JOB.scp" \
                --dumpdir "${dumpdir}/${name}/raw/dump.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished feature extraction of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished feature extraction."
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Statistics computation (optional) and normalization"

    if [ ${norm_name} == "self" ]; then
        # calculate statistics for normalization

        # src
        name="${srcspk}_${train_set}"
        echo "Statistics computation start. See the progress via ${dumpdir}/${name}/compute_statistics_${src_feat}.log."
        ${train_cmd} "${dumpdir}/${name}/compute_statistics_${src_feat}.log" \
            compute_statistics.py \
                --config "${conf}" \
                --rootdir "${dumpdir}/${name}/raw" \
                --dumpdir "${dumpdir}/${name}" \
                --feat_type "${src_feat}" \
                --verbose "${verbose}"

        # trg
        name="${trgspk}_${train_set}"
        echo "Statistics computation start. See the progress via ${dumpdir}/${name}/compute_statistics_${trg_feat}.log."
        ${train_cmd} "${dumpdir}/${name}/compute_statistics_${trg_feat}.log" \
            compute_statistics.py \
                --config "${conf}" \
                --rootdir "${dumpdir}/${name}/raw" \
                --dumpdir "${dumpdir}/${name}" \
                --feat_type "${trg_feat}" \
                --verbose "${verbose}"
    fi

    # if norm_name=self, then use $conf; else use config in pretrained_model_dir
    if [ ${norm_name} == "self" ]; then
        config_for_feature_extraction="${conf}"
    else
        config_for_feature_extraction="${pretrained_model_dir}/config.yml"
    fi

    # normalize and dump them
    # src
    spk="${srcspk}"
    for name in "${spk}_${train_set}" "${spk}_${dev_set}" "${spk}_${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm_${norm_name}" ] && mkdir -p "${dumpdir}/${name}/norm_${norm_name}"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm_${norm_name}/normalize_${src_feat}.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm_${norm_name}/normalize_${src_feat}.JOB.log" \
            normalize.py \
                --config "${config_for_feature_extraction}" \
                --stats "${src_stats}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --verbose "${verbose}" \
                --feat_type "${src_feat}" \
                --skip-wav-copy
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished ${spk} side normalization."

    # trg
    spk="${trgspk}"
    for name in "${spk}_${train_set}" "${spk}_${dev_set}" "${spk}_${eval_set}"; do
    (
        [ ! -e "${dumpdir}/${name}/norm_${norm_name}" ] && mkdir -p "${dumpdir}/${name}/norm_${norm_name}"
        echo "Nomalization start. See the progress via ${dumpdir}/${name}/norm_${norm_name}/normalize_${trg_feat}.*.log."
        ${train_cmd} JOB=1:${n_jobs} "${dumpdir}/${name}/norm_${norm_name}/normalize_${trg_feat}.JOB.log" \
            normalize.py \
                --config "${config_for_feature_extraction}" \
                --stats "${trg_stats}" \
                --rootdir "${dumpdir}/${name}/raw/dump.JOB" \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --verbose "${verbose}" \
                --feat_type "${trg_feat}" \
                --skip-wav-copy
        echo "Successfully finished normalization of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished ${spk} side normalization."
fi

if [ -z ${tag} ]; then
    expname=${srcspk}_${trgspk}_$(basename ${conf%.*})
else
    expname=${srcspk}_${trgspk}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        echo "Not Implemented yet. Usually VC training using arctic can be done with 1 GPU."
        exit 1
    fi

    if [ ! -z ${pretrained_model_checkpoint} ]; then
        echo "Pretraining not Implemented yet."
        exit 1
    else
        cp "${dumpdir}/${trgspk}_train/stats.${stats_ext}" "${expdir}/"
        echo "Training start. See the progress via ${expdir}/train.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
            vc_train.py \
                --config "${conf}" \
                --src-train-dumpdir "${dumpdir}/${srcspk}_train/norm_${norm_name}" \
                --src-dev-dumpdir "${dumpdir}/${srcspk}_dev/norm_${norm_name}" \
                --src-feat-type "${src_feat}" \
                --trg-train-dumpdir "${dumpdir}/${trgspk}_train/norm_${norm_name}" \
                --trg-dev-dumpdir "${dumpdir}/${trgspk}_dev/norm_${norm_name}" \
                --trg-feat-type "${trg_feat}" \
                --trg-stats "${expdir}/stats.${stats_ext}" \
                --train-dp-input-dir "${dumpdir}/${srcspk}_train/norm_${norm_name}" \
                --dev-dp-input-dir "${dumpdir}/${srcspk}_dev/norm_${norm_name}" \
                --train-duration-dir "${train_duration_dir}" \
                --dev-duration-dir "${dev_duration_dir}" \
                --outdir "${expdir}" \
                --resume "${resume}" \
                --verbose "${verbose}"
    fi
    echo "Successfully finished training."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${srcspk}_test"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.*.log."
        CUDA_VISIBLE_DEVICES="" ${cuda_cmd} JOB=1:${n_jobs} --gpu 0 "${outdir}/${name}/decode.JOB.log" \
            vc_decode.py \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --dp_input_dumpdir "${dumpdir}/${name}/norm_${norm_name}/dump.JOB" \
                --checkpoint "${checkpoint}" \
                --src-feat-type "${src_feat}" \
                --trg-feat-type "${trg_feat}" \
                --trg-stats "${expdir}/stats.${stats_ext}" \
                --outdir "${outdir}/${name}/out.JOB" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Successfully finished decoding."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for _set in "test"; do
        name="${srcspk}_${_set}"
        echo "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            local/evaluate.py \
                --wavdir "${outdir}/${name}" \
                --data_root "${db_root}/${trgspk}" \
                --set_name ${_set} \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml"
        grep "Mean MCD" "${outdir}/${name}/evaluation.log"
    done
fi
