import argparse
import h5py
import os
from tqdm import tqdm

from speechbrain.pretrained import EncoderClassifier
import torchaudio

def read_wav_scp(wav_scp_path):
    """Reads a Kaldi-style wav.scp file."""
    wavs = []
    with open(wav_scp_path, "r") as f:
        for line in f:
            utt_id, _, wav_path = line.strip().split()[:3]
            wavs.append((utt_id, wav_path))
    return wavs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", required=True, help="Path to wav.scp file")
    parser.add_argument("--output_dir", required=True, help="Output directory for embedding H5 files")
    parser.add_argument("--model_source", default="speechbrain/spkrec-ecapa-voxceleb", help="SpeechBrain model source")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    classifier = EncoderClassifier.from_hparams(source=args.model_source, run_opts={"device": "cuda"})

    wavs = read_wav_scp(args.wav_scp)
    for utt_id, wav_path in tqdm(wavs, desc="Extracting embeddings"):
        signal, fs = torchaudio.load(wav_path)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        emb = classifier.encode_batch(signal).squeeze().cpu().numpy()
        
        # Save to separate H5 file for each utterance
        output_file = os.path.join(args.output_dir, f"{utt_id}.h5")
        with h5py.File(output_file, "w") as h5f:
            h5f.create_dataset("embedding", data=emb)
            
        if args.verbose > 0:
            print(f"Saved embedding for {utt_id} to {output_file}")

if __name__ == "__main__":
    main()