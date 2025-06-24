import os
import random


from datasets import load_dataset
import soundfile as sf  # For saving .wav files
import os
import argparse

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install the 'datasets' library: pip install datasets")


def write_wav_scp(audio_files, output_path, fs=16000):
    """Write a Kaldi-style wav.scp file from the list of audio files."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i, audio_file in enumerate(audio_files):
            utt_id = audio_file[0]
            # Write the line in the format:${id} cat ${filename} | sox -t wav - -c 1 -b 16 -t wav - rate ${fs} |
            f.write(f"{utt_id} sox {audio_file[1]} -c 1 -b 16 -t wav - rate {fs} |\n")

def write_text_file(utterances, output_path):
    """Write a text file with utterances."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for uttid, _,  text in utterances:
            f.write(f"{uttid}|{text}\n")



def main():

    """Load a dataset using the Hugging Face datasets library."""

    parser = argparse.ArgumentParser(description="Prepare wav.scp for Kaldi-style data directory.")
    parser.add_argument("--name", default="sqrk/mixat-tri", help="Name of the dataset to load")
    parser.add_argument("--data_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output_dir", required=True, help="Output path for wav.scp file")
    parser.add_argument("--fs", type=int, default=16000, help="Sample rate for audio files")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    args = parser.parse_args()
    
    dataset = load_dataset(args.name, split='test', cache_dir=args.data_dir)

    wav_dir = os.path.join(args.data_dir, "female_ag")
    text_dir = os.path.join(args.data_dir, "test.txt")
    os.makedirs(wav_dir, exist_ok=True)

    transcripts = []
    # 3. Iterate and save audio as .wav files
    for i, item in enumerate(dataset):

        # audio = item["audio"] 
        
        output_path = os.path.join(wav_dir, f"female_ag_{i}.wav")
        

        # sf.write(output_path, audio["array"], audio["sampling_rate"])
        # if args.verbose>0:
        #     if i % 100 == 0:
        #         print(f"Saved {i} files...")

        uttid = os.path.splitext(os.path.basename(output_path))[0]
        transcripts.append((uttid, output_path, item["transcript"]))

    print("All audio files saved.")

    # save the transcripts to a text file
    write_text_file(transcripts, text_dir)
    write_wav_scp(transcripts, os.path.join(args.output_dir, "female_ag_test", "wav.scp"), fs=args.fs)

    # clean up the cache and parquet files
    # import shutil
    # datadir = os.path.join(data_dir, 'data')
    # cachedir = os.path.join(data_dir, '.cache')
    # shutil.rmtree(datadir, ignore_errors=True)
    # shutil.rmtree(cachedir, ignore_errors=True)



if __name__ == "__main__":
    main()