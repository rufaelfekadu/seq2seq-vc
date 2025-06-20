import os
import random


def find_audio(path):
    """Recursively find all .wav files in the given path."""
    audio_files = {}
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                uttid = os.path.splitext(file)[0]
                audio_files[uttid] = os.path.join(root, file)
    return audio_files

def write_wav_scp(audio_files, output_path, fs=16000):
    """Write a Kaldi-style wav.scp file from the list of audio files."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i, audio_file in enumerate(audio_files):
            utt_id = os.path.splitext(os.path.basename(audio_file))[0]
            # Write the line in the format:${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |
            f.write(f"{utt_id} sox {audio_file} -c 1 -b 16 -t wav - rate {fs} |\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare wav.scp for Kaldi-style data directory.")
    parser.add_argument("--data_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--srcspks", required=True, nargs='+', help="Source speakers, space-separated")
    parser.add_argument("--tgtspk", default="ar-XA-Wavenet-A", help="Target speaker name for the data directory")
    parser.add_argument("--oodspks", default="", help="Out-of-distribution speakers, space-separated")
    parser.add_argument("--output_dir", required=True, help="Output path for wav.scp file")
    parser.add_argument("--num_dev", type=int, default=20, help="Number of utterances for development set")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the data before splitting")
    parser.add_argument("--fs", type=int, default=16000, help="Sample rate for audio files")
    parser.add_argument("--train_set", default="train_nodev", help="Name of training data directory")
    parser.add_argument("--dev_set", default="dev", help="Name of development data directory")
    parser.add_argument("--eval_set", default="test", help="Name of evaluation data directory")
    
    args = parser.parse_args()

    tgt_train_audio_files = []
    tgt_dev_audio_files = []
    tgt_eval_audio_files = []


    print(f"Preparing data for speakers: {args.srcspks} with target speaker: {args.tgtspk} and OOD speakers: {args.oodspks}")
    for srcspk in args.srcspks:
        
        spk_dir = os.path.join(args.data_dir, f"ArVoice_syn-{srcspk}", f"{srcspk}")
        audio_files = find_audio(spk_dir)
        tgt_spk_dir = os.path.join(args.data_dir, f"ArVoice_syn-{srcspk}", f"{args.tgtspk}")
        tgt_audio_files = find_audio(tgt_spk_dir)
        
        if not audio_files or not tgt_audio_files:
            print(f"No audio files found for speaker {srcspk} in {spk_dir}.")
            continue

        uttids = sorted(list(audio_files.keys()))

        # split into train, dev, eval
        eval_uttids = [uttid for uttid in uttids if "test" in uttid]
        train_uttids = [uttid for uttid in uttids if uttid not in eval_uttids]
        
        # if shuffle, shuffle the train set
        if args.shuffle:
            random.shuffle(train_uttids)

        dev_uttids = train_uttids[:args.num_dev]
        train_uttids = train_uttids[args.num_dev:]

        # select audio files for each set
        train_audio_files = [audio_files[uttid] for uttid in train_uttids]
        dev_audio_files = [audio_files[uttid] for uttid in dev_uttids]
        eval_audio_files = [audio_files[uttid] for uttid in eval_uttids]

        # select target speaker audio files
        tgt_train_audio_files += [tgt_audio_files[uttid] for uttid in train_uttids]
        tgt_dev_audio_files += [tgt_audio_files[uttid] for uttid in dev_uttids]
        tgt_eval_audio_files += [tgt_audio_files[uttid] for uttid in eval_uttids]                           
        
        # Write wav.scp files for each set
        write_wav_scp(train_audio_files, os.path.join(args.output_dir, f"{srcspk}_train", "wav.scp"), args.fs)
        write_wav_scp(dev_audio_files, os.path.join(args.output_dir, f"{srcspk}_dev", "wav.scp"), args.fs)
        write_wav_scp(eval_audio_files, os.path.join(args.output_dir, f"{srcspk}_test", "wav.scp"), args.fs)

    # Write target speaker wav.scp files
    write_wav_scp(tgt_train_audio_files, os.path.join(args.output_dir, f"{args.tgtspk}_train", "wav.scp"), args.fs)
    write_wav_scp(tgt_dev_audio_files, os.path.join(args.output_dir, f"{args.tgtspk}_dev", "wav.scp"), args.fs)
    write_wav_scp(tgt_eval_audio_files, os.path.join(args.output_dir, f"{args.tgtspk}_test", "wav.scp"), args.fs)

    # prepare oodspks if any
    if args.oodspks:
        oodspks = args.oodspks.split()
        for oodspk in oodspks:
            ood_spk_dir = os.path.join(args.data_dir, f"ArVoice_syn-{oodspk}", f"{oodspk}")
            ood_audio_files = find_audio(ood_spk_dir)
            if not ood_audio_files:
                print(f"No audio files found for OOD speaker {oodspk} in {ood_spk_dir}.")
                continue
            
            write_wav_scp(list(ood_audio_files.values()), os.path.join(args.output_dir, f"{oodspk}_test", "wav.scp"), args.fs)

if __name__ == "__main__":
    main()