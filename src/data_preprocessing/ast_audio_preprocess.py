from argparse import ArgumentParser
import numpy as np
import librosa
import os
from tqdm import tqdm
import ffmpeg
import subprocess
from transformers import ASTFeatureExtractor
import concurrent.futures


ACCEPTED_AUDIO_EXTENSIONS = ['.m4a', '.wav']

parser = ArgumentParser(description="Raw audio to spectogram conversion")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the directory where audio files are located")

parser.add_argument("--save-dir", required=True, type=str,
                    help="Absolute path to the directory where spectrograms will be saved")

parser.add_argument("--audio-length", default=10.26, type=float,
                    help="length of the audio that will be converted to spectogram. Longer audio \
                          files will be clipped, and shorter files will be repeated")

parser.add_argument("--sampling-rate", default=16000, type=int,
                    help="The number of samples per second of audio (for AST it must be 16KHz)")

parser.add_argument("--mean", default=-4.2677393, type=float,
                    help="Mean value of the audio files (calculated on the training set)")

parser.add_argument("--std", default=4.5689974, type=float,
                    help="Standard deviation value of the audio files (calculated on the training set)")

parser.add_argument("--num-threads", default=12, type=int,
                    help="Number of threads to process audio")


def normalize_audio(file_path, root, file_base):
    output_path = os.path.join(root, f"{file_base}_normalized.wav")
    ffmpeg.input(file_path).output(output_path, ar=16000, acodec="pcm_s16le", ac=1, loglevel="quiet").run()
    normalize_command = ["ffmpeg-normalize", output_path, "--normalization-type", "rms", "-t", "-23", "-o", output_path, '-f']
    subprocess.run(normalize_command)
    return output_path

def stretch_audio(args, waveform):
    duration = librosa.get_duration(y=waveform, sr=args.sampling_rate)
    if duration < args.audio_length:
        padding_length = np.round((args.audio_length * args.sampling_rate - duration * args.sampling_rate))
        padding =  waveform[:int(padding_length)]
        while padding_length > 0:
          waveform = np.concatenate((waveform, padding))
          padding_length -= padding.shape[0]
    return waveform[:int(args.audio_length * args.sampling_rate)]


def save_processed_data(args, root, file_base, data):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    file_base += ".npy"
    new_file_path = os.path.join(save_dir, file_base)
    np.save(new_file_path, data)

def process_audio(args, feature_extractor, file_path, root, file_base):
    normalized_file_path = normalize_audio(file_path, root, file_base)
    waveform, _ = librosa.load(normalized_file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
    waveform = stretch_audio(args, waveform)
    inputs = feature_extractor(waveform, sampling_rate=args.sampling_rate, padding="max_length", return_tensors="np")
    input_values = inputs.input_values
    os.remove(normalized_file_path)
    save_processed_data(args, root, file_base, input_values)

def main():
    args = parser.parse_args()
    assert args.sampling_rate == 16000, f"For AST sampling rate should be equal to 16KHz, if you are sure \
        you want to use other value for sampling rate delete this assert (current sampling rate is {args.sampling_rate})"

    feature_extractor = ASTFeatureExtractor(sampling_rate=args.sampling_rate, mean=args.mean, std=args.std)

    roots = []
    file_paths = []
    file_bases = []
    for root, _, files in tqdm(os.walk(args.data_dir), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            file_path = os.path.join(root, file_name)
            file_base, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_AUDIO_EXTENSIONS:
                continue
            roots.append(root)
            file_paths.append(file_path)
            file_bases.append(file_base)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        tasks = [executor.submit(process_audio, args, feature_extractor, file_path, root, file_base) for (file_path, root, file_base) in zip(file_paths, roots, file_bases)]

if __name__ == "__main__":
    main()
