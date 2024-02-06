from argparse import ArgumentParser
import numpy as np
import librosa
import os
from tqdm import tqdm
import ffmpeg
import concurrent.futures


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
    return output_path

def stretch_audio(args, waveform):
    duration = librosa.get_duration(y=waveform, sr=args.sampling_rate)
    if duration < 6:
        padding_length = np.round((args.audio_length - duration) * args.sampling_rate)
        padded_waveform = waveform[:int(padding_length)]
        waveform = np.concatenate((waveform, padded_waveform))
    return waveform


def get_mean_std(args, file_path, root, file_base):
    normalized_file_path = normalize_audio(file_path, root, file_base)
    waveform, _ = librosa.load(normalized_file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
    waveform = stretch_audio(args, waveform)
    os.remove(normalized_file_path)
    mean = np.mean(waveform)
    std = np.std(waveform)
    return mean, std


def main():
    args = parser.parse_args()



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
        tasks = [executor.submit(get_mean_std, args, file_path, root, file_base) for (file_path, root, file_base) in zip(file_paths, roots, file_bases)]
        for task in concurrent.futures.as_completed(tasks):
            mean, std = task.result()
            means.append(mean)
            stds.append(std)

    print(np.mean(means), np.mean(stds))


if __name__ == "__main__":
    main()
