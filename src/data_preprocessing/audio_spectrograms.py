from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from tqdm import tqdm
import ffmpeg
import subprocess
import concurrent.futures


ACCEPTED_AUDIO_EXTENSIONS = ['.m4a', '.wav']

parser = ArgumentParser(description="Raw audio to spectogram conversion")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the directory where audio files are located")

parser.add_argument("--save-dir", required=True, type=str,
                    help="Absolute path to the directory where spectrograms will be saved")

parser.add_argument("--audio-length", default=6, type=int,
                    help="length of the audio that will be converted to spectogram. Longer audio \
                          files will be clipped, and shorter files will be repeated")

parser.add_argument("--sampling-rate", default=16000, type=int,
                    help="The number of samples per second of audio")

parser.add_argument("--hop-length", default=10, type=int,
                    help="Period of time between adjacent STFT columns specified in milliseconds")

parser.add_argument("--window-length", default=25, type=int,
                    help="Length of the window of STFT columns ")

parser.add_argument("--n-fft", default=512, type=int,
                    help="Length of the windowed signal after padding with zeros")

parser.add_argument("--power", default=0.3, type=float,
                    help="power applied during power-law compression")

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


def compute_spectrograms(args, waveform):
    hop_length = int(args.hop_length / 1000 * args.sampling_rate)
    window_length = int(args.window_length / 1000 * args.sampling_rate)
    return librosa.stft(waveform, n_fft=args.n_fft, hop_length=hop_length,
                        win_length=window_length, window="hann")

def power_law_compression(args, spectrogram):
    magnitude = np.abs(spectrogram)
    phase = np.angle(spectrogram)
    compressed_magnitude = np.sign(magnitude) * np.abs(magnitude) ** args.power
    compressed_phase = np.sign(phase) * np.abs(phase) ** args.power
    compressed_spectrogram = np.stack((compressed_magnitude, compressed_phase), axis=0)
    return compressed_spectrogram


def save_spectrogram(args, root, file_base, data):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    file_base += ".npy"
    new_file_path = os.path.join(save_dir, file_base)
    np.save(new_file_path, data)


def visualize_spectrogram(args, spectrogram):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(abs(spectrogram), ref=np.max), sr=args.sampling_rate, hop_length=args.hop_length, y_axis='linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Magnitude Spectrogram')
    plt.tight_layout()
    plt.show()


def process_audio(args, file_path, root, file_base):
    normalized_file_path = normalize_audio(file_path, root, file_base)
    waveform, _ = librosa.load(normalized_file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
    waveform = stretch_audio(args, waveform)
    spectrogram = compute_spectrograms(args, waveform)
    spectrogram = power_law_compression(args, spectrogram)
    os.remove(normalized_file_path)
    save_spectrogram(args, root, file_base, spectrogram)


def main():
    args = parser.parse_args()

    roots = []
    file_paths = []
    file_bases = []
    for root, _, files in tqdm(os.walk(args.data_dir), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            file_path = os.path.join(root, file_name)
            file_base, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_AUDIO_EXTENSIONS:
                continue

            if int(file_base) <= 20:
                roots.append(root)
                file_paths.append(file_path)
                file_bases.append(file_base)
                

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        tasks = [executor.submit(process_audio, args, file_path, root, file_base) for (file_path, root, file_base) in zip(file_paths, roots, file_bases)]

if __name__ == "__main__":
    main()
