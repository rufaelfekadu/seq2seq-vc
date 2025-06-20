#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import multiprocessing as mp
import os

import numpy as np
import librosa

import torch
import torchaudio
from tqdm import tqdm
import yaml

from seq2seq_vc.utils import find_files
from seq2seq_vc.utils.types import str2bool
from seq2seq_vc.evaluate.dtw_based import calculate_mcd_f0
from seq2seq_vc.evaluate.asr import load_asr_model, transcribe, calculate_measures

from transformers import WhisperProcessor, WhisperForConditionalGeneration, SpeechT5ForSpeechToText, SpeechT5Processor, SpeechT5Tokenizer
import jiwer
import re
import pyarabic.araby as araby

from speechbrain.inference import EncoderClassifier, SpeakerRecognition
from speechbrain.utils.metric_stats import BinaryMetricStats
from speechbrain.utils.metric_stats import EER


ASR_PRETRAINED_MODEL = "clu-ling/whisper-large-v2-arabic-5k-steps"
SPEECHT5_PRETRAINED_MODEL = "MBZUAI/artst_asr"

def load_speecht5_model(device):
    processor = SpeechT5Processor.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    model = SpeechT5ForSpeechToText.from_pretrained(SPEECHT5_PRETRAINED_MODEL).to(
        device
    )
    tokenizer = SpeechT5Tokenizer.from_pretrained(SPEECHT5_PRETRAINED_MODEL)
    models = {"model": model, "processor": processor, "tokenizer": tokenizer}
    return models

def load_asr_model(device):
    """Load model"""
    print(f"[INFO]: Load the pre-trained ASR by {ASR_PRETRAINED_MODEL}.")
    processor = WhisperProcessor.from_pretrained(ASR_PRETRAINED_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(ASR_PRETRAINED_MODEL).to(
        device
    )
    models = {"model": model, "processor": processor}
    return models

def clean_text(text):
  """Normalizes TRANSCRIPT"""
  text = re.sub(r'[\,\?\.\!\-\;\:\"\“\%\٪\‘\”\�\«\»\،\.\:\؟\؛\*\>\<]', '', text) + " " # special characters
  text = re.sub(r'http\S+', '', text) + " " # links
  text = re.sub(r'[\[\]\(\)\-\/\{\}]', '', text) + " " # brackets
  text = re.sub(r'\s+', ' ', text) + " " # extra white space
  text = araby.strip_diacritics(text) # remove diacrirics
  return text.strip()

def normalize_sentence(sentence):
    """Normalize sentence"""
    sentence = clean_text(sentence)
    return sentence

def transcribe(model, device, wav):
    """Calculate score on one single waveform"""
    # preparation

    inputs = model["processor"](
        audio=wav, sampling_rate=16000, return_tensors="pt"
    )
    inputs = inputs.to(device)

    # forward
    predicted_ids = model["model"].generate(**inputs)
    transcription = model["processor"].batch_decode(
        predicted_ids, skip_special_tokens=True
    )

    return transcription

def calculate_measures(groundtruth, transcription):
    """Calculate character/word measures (hits, subs, inserts, deletes) for one given sentence"""
    groundtruth = normalize_sentence(groundtruth)
    transcription = normalize_sentence(transcription)

    c_result = jiwer.cer(groundtruth, transcription, return_dict=True)
    w_result = jiwer.compute_measures(groundtruth, transcription)

    return c_result, w_result, groundtruth, transcription

def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]

def get_speaker(path):
    return os.path.dirname(path).split("/")[-1] 

def compute_eer(trg_spk, source_root, gt_root, converted_files):
    
    eer_calculator = BinaryMetricStats()
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="downloads/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
    
    ref_files = find_files(gt_root, query="*.wav")
    ref_files = [f for f in ref_files if "test" in f]

    source_files = find_files(source_root, query="*.wav")
    source_files = [f for f in source_files if "test" in f]

    # other_spk_files = find_files(other_spk_dir, query="*.wav")
    # other_spk_files = [f for f in other_spk_files if "test" in f]

    # # filter out the target speaker files
    # other_spk_files = [f for f in other_spk_files if trg_spk != get_speaker(f)]
    # other_tgt_files = [f for f in other_spk_files if trg_spk == get_speaker(f)]

    # if len(other_spk_files) > len(converted_files):
    #     print(f"Warning: There are more other speaker files ({len(other_spk_files)}) than converted files ({len(converted_files)}). Randomly selecting {len(converted_files)} other speaker files.")
    #     other_spk_files = np.random.choice(other_spk_files, size=len(converted_files), replace=False).tolist()
    print(f"Number of reference files: {len(ref_files)}, source files: {len(source_files)}, converted files: {len(converted_files)}")
    pairs_gen = []
    for i, cv_path in enumerate(converted_files):
        basename = get_basename(cv_path)
        
        # negative pairs gen gt
        for k, ref_file in enumerate(ref_files):
            if basename == get_basename(ref_file): 
                continue
            pairs_gen.append((cv_path, ref_file, 0))

    pairs_gt = []
    for i, ref_path in enumerate(ref_files):
        basename = get_basename(ref_path)

        for j, ref_path2 in enumerate(ref_files):
            if get_basename(ref_path2) == basename:
                continue
            pairs_gt.append((ref_path2, ref_path, 1))

        # negative pairs gen other
        for k, source_file in enumerate(source_files):
            if basename == get_basename(source_file): 
                continue
            pairs_gt.append((source_file, ref_path, 0))
     
    scores_gen = []
    for i, (cv_path, ref_path, label) in enumerate(tqdm(pairs_gen, desc="Computing EER")):
        p_score, p_pred = verification.verify_files(ref_path, cv_path)
        
        scores_gen.append([f"{get_basename(cv_path)}_{get_basename(ref_path)}", p_score.cpu().item(), p_pred.cpu().item(), label])
        eer_calculator.append(
            [f"{get_basename(cv_path)}_{get_basename(ref_path)}"],
            p_score.cpu(),
            torch.tensor([label])
        )
    np.save(f"downloads/scores_gen_{trg_spk}.npy", np.array(scores_gen))
    eer_gen = eer_calculator.summarize()
    
    eer_calculator.clear()
    scores_gt = []
    for i, (ref_path2, ref_path, label) in enumerate(tqdm(pairs_gt, desc="Computing EER GT")):
        p_score, p_pred = verification.verify_files(ref_path2, ref_path)
        scores_gt.append([p_score.item(), label])

        eer_calculator.append(
            [f"{get_basename(ref_path2)}_{get_basename(ref_path)}"],
            torch.tensor([p_score.item()]),
            torch.tensor([label]),
        )
    # save the scores and labels for later use
    np.save(f"downloads/scores_gt_{trg_spk}.npy", np.array(scores_gt))
    eer_gt = eer_calculator.summarize()

    return eer_gen, eer_gt

def _calculate_asr_score(model, device, file_list, groundtruths):
    keys = ["hits", "substitutions", "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        groundtruth = groundtruths[basename]  # get rid of the first character "E"

        # load waveform
        wav, _ = librosa.load(cvt_wav_path, sr=16000)

        # trascribe
        transcription = transcribe(model, device, wav)
        transcription = "".join(str(i) for i in transcription)
        transcription = transcription.replace(" ", "")

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(
            groundtruth, transcription
        )

        ers[basename] = [
            c_result["cer"] * 100.0,
            w_result["wer"] * 100.0,
            norm_transcription,
            norm_groundtruth,
        ]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]

    # calculate over whole set
    def er(r):
        return (
            float(r["substitutions"] + r["deletions"] + r["insertions"])
            / float(r["substitutions"] + r["deletions"] + r["hits"])
            * 100.0
        )

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer

def _calculate_mcd_f0(file_list, gt_root, segments, trgspk, f0min, f0max, results, gv=False):
    for i, cvt_wav_path in enumerate(file_list):
        basename = get_basename(cvt_wav_path)
        
        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, basename + ".wav")

        # read both converted and ground truth wav
        cvt_wav, cvt_fs = librosa.load(cvt_wav_path, sr=None)
        if segments is not None:
            gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None,
                                         offset=segments[basename]["offset"],
                                         duration=segments[basename]["duration"]
                                         )
        else:
            gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None)
        if cvt_fs != gt_fs:
            cvt_wav = torchaudio.transforms.Resample(cvt_fs, gt_fs)(torch.from_numpy(cvt_wav)).numpy()

        # calculate MCD, F0RMSE, F0CORR and DDUR
        res = calculate_mcd_f0(cvt_wav, gt_wav, gt_fs, f0min, f0max, calculate_gv=gv)

        results.append([basename, res])

def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument("--wavdir", required=True, type=str, help="directory for converted waveforms")
    parser.add_argument("--trgspk", required=True, type=str, help="target speaker")
    parser.add_argument("--set_name", required=True, type=str, help="set name (to retrive gt text)")
    parser.add_argument("--data_root", type=str, default="./data", help="directory of data")
    parser.add_argument("--src_root", type=str, default=None, help="directory of source data")
    parser.add_argument("--other_spk_dir", type=str, default=None, help="directory of other speakers")
    parser.add_argument("--segments", type=str, default=None, help="segments file")
    parser.add_argument("--f0_path", required=True, type=str, help="yaml file storing f0 ranges")
    parser.add_argument("--n_jobs", default=10, type=int, help="number of parallel jobs")
    return parser

def main():
    args = get_parser().parse_args()

    trgspk = args.trgspk
    gt_root = args.data_root
    transcription_path = os.path.join(args.data_root, f"{args.set_name}.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # load f0min and f0 max
    with open(args.f0_path, 'r') as f:
        f0_all = yaml.load(f, Loader=yaml.FullLoader)
    f0min = f0_all[trgspk]["f0min"]
    f0max = f0_all[trgspk]["f0max"]

    # load ground truth transcriptions
    with open(transcription_path, "r") as f:
        lines = f.read().splitlines()
    groundtruths = {line.split("|")[0]: line.split("|")[1] for line in lines}
    # load segments if provided
    if args.segments is not None:
        with open(args.segments, "r") as f:
            lines = f.read().splitlines()
        segments = {}
        for line in lines:
            _id, _, start, end = line.split(" ")
            segments[_id] = {
                "offset": float(start),
                "duration": float(end) - float(start)
            }
    else:
        segments = None

    # find converted files
    converted_files = sorted(find_files(args.wavdir, query="*.wav"))
    print("number of utterances = {}".format(len(converted_files)))

    ##############################

    print("Calculating ASR-based score...")

    # load ASR model
    asr_model = load_asr_model(device)

    # calculate error rates
    ers, cer, wer = _calculate_asr_score(
        asr_model, device, converted_files, groundtruths
    )
    
    ##############################

    print("Calculating EER...")
    import json
    eer_gen, eer_gt = compute_eer(trgspk, args.src_root, gt_root, converted_files)
    with open(f"downloads/eer_{trgspk}.txt", "w") as f:
        # dump as json
        json.dump({
            "eer_gen": eer_gen,
            "eer_gt": eer_gt
        }, f, indent=4)

    print({"eer_gen": eer_gen, "eer_gt": eer_gt})

    ##############################

    print("Calculating MCD and f0-related scores...")
    # Get and divide list
    file_lists = np.array_split(converted_files, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        results = manager.list()
        processes = []
        for f in file_lists:
            p = mp.Process(
                target=_calculate_mcd_f0,
                args=(f, gt_root, segments, trgspk, f0min, f0max, results, False),
            )
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        sorted_results = sorted(results, key=lambda x:x[0])
        results = []
        for result in sorted_results:
            d = {k: v for k, v in result[1].items()}
            d["basename"] = result[0]
            d["CER"] = ers[result[0]][0]
            d["GT_TRANSCRIPTION"] = ers[result[0]][2]
            d["CV_TRANSCRIPTION"] = ers[result[0]][3]
            results.append(d)
        
    # utterance wise result
    for result in results:
        print(
            "{} {:.2f} {:.2f} {:.2f} {:.2f} {:.1f} \t{} | {}".format(
                result["basename"],
                result["MCD"],
                result["F0RMSE"],
                result["F0CORR"],
                result["DDUR"],
                result["CER"],
                result["GT_TRANSCRIPTION"],
                result["CV_TRANSCRIPTION"],
            )
        )

    # average result
    mMCD = np.mean(np.array([result["MCD"] for result in results]))
    mf0RMSE = np.mean(np.array([result["F0RMSE"] for result in results]))
    mf0CORR = np.mean(np.array([result["F0CORR"] for result in results]))
    mDDUR = np.mean(np.array([result["DDUR"] for result in results]))
    mCER = cer
    mEER = eer_gen["EER"] 

    print(
        "Mean MCD, f0RMSE, f0CORR, DDUR, CER: {:.2f} {:.2f} {:.3f} {:.3f} {:.1f}".format(
            mMCD, mf0RMSE, mf0CORR, mDDUR, mCER
        )
    )

    
    

if __name__ == "__main__":
    main()