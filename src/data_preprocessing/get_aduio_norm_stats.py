from argparse import ArgumentParser
import numpy as np
import librosa
import os
from tqdm import tqdm
import ffmpeg
import concurrent.futures
import subprocess
from transformers import ASTFeatureExtractor


ACCEPTED_AUDIO_EXTENSIONS = ['.m4a', '.wav']

parser = ArgumentParser(description="Raw audio to spectogram conversion")

parser.add_argument("--train-data-dir", required=True, type=str,
                    help="Absolute path to the directory where audio files are located (for train dataset)")

parser.add_argument("--audio-length", default=6, type=int,
                    help="length of the audio that will be converted to spectogram. Longer audio \
                          files will be clipped, and shorter files will be repeated")

parser.add_argument("--sampling-rate", default=16000, type=int,
                    help="The number of samples per second of audio")


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

def get_mean_std(args, file_path, root, file_base, feature_extractor):
    normalized_file_path = normalize_audio(file_path, root, file_base)
    waveform, _ = librosa.load(normalized_file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
    waveform = stretch_audio(args, waveform)
    os.remove(normalized_file_path)

    inputs = feature_extractor(waveform, sampling_rate=args.sampling_rate, padding="max_length", return_tensors="np")
    input_values = inputs.input_values
    cur_mean = np.mean(input_values)
    cur_std = np.std(input_values)

    return cur_mean, cur_std


def main():
    args = parser.parse_args()

    feature_extractor = ASTFeatureExtractor(sampling_rate=args.sampling_rate, do_normalize=False, max_length=1000)

    roots = []
    file_paths = []
    file_bases = []
    for root, _, files in tqdm(os.walk(args.train_data_dir), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            file_path = os.path.join(root, file_name)
            file_base, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_AUDIO_EXTENSIONS:
                continue
            roots.append(root)
            file_paths.append(file_path)
            file_bases.append(file_base)

    means = []
    stds = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        tasks = [executor.submit(get_mean_std, args, file_path, root, file_base, feature_extractor) for (file_path, root, file_base) in zip(file_paths, roots, file_bases)]
        for task in concurrent.futures.as_completed(tasks):
            mean, std = task.result()
            means.append(mean)
            stds.append(std)

    print(np.mean(means), np.mean(stds))
    print(len(means), len(stds))

    with open("statistics.txt", "w") as file:
        file.write(f"Mean: {np.mean(means)}\n")
        file.write(f"Stds: {np.mean(stds)}\n")


if __name__ == "__main__":
    main()
