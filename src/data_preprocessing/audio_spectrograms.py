from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
from tqdm import tqdm


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


def stretch_audio(args, waveform):
    duration = librosa.get_duration(y=waveform, sr=args.sampling_rate)
    if duration < 6:
        padding_length = np.round((args.audio_length - duration) * args.sampling_rate)
        padded_waveform = waveform[:int(padding_length)]
        waveform = np.concatenate((waveform, padded_waveform))
    return waveform


def compute_spectrograms(args, waveform):
    hop_length = int(args.hop_length / 1000 * args.sampling_rate)
    window_length = int(args.window_length / 1000 * args.sampling_rate)
    return librosa.stft(waveform, n_fft=args.n_fft, hop_length=hop_length,
                        win_length=window_length, window="hann")


def power_law_compression(args, spectrogram):
    real = np.real(spectrogram)
    imag = np.imag(spectrogram)
    compressed_real = np.sign(real) * np.abs(real) ** args.power
    compressed_imag = np.sign(imag) * np.abs(imag) ** args.power
    compressed_spectrogram = np.stack((compressed_real, compressed_imag), axis=0)
    return compressed_spectrogram


def save_spectrogram(args, root, file_name, spectrogram):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    name, _ = os.path.splitext(file_name)
    name += ".npy"
    new_file_path = os.path.join(save_dir, name)
    np.save(new_file_path, spectrogram)


def visualize_spectrogram(args, spectrogram):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(abs(spectrogram), ref=np.max), sr=args.sampling_rate, hop_length=args.hop_length, y_axis='linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Magnitude Spectrogram')
    plt.tight_layout()
    plt.show()


def main():
    args = parser.parse_args()

    for root, _, files in tqdm(os.walk(args.data_dir), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            _, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_AUDIO_EXTENSIONS:
                continue
            file_path = os.path.join(root, file_name)
            waveform, _ = librosa.load(file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
            waveform = stretch_audio(args, waveform)
            spectrogram = compute_spectrograms(args, waveform)
            # visualize_spectrogram(args, spectrogram)
            spectrogram = power_law_compression(args, spectrogram)
            save_spectrogram(args, root, file_name, spectrogram)


if __name__ == "__main__":
    main()
