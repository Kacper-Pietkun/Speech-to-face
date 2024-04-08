import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from argparse import ArgumentParser
from models.voice_encoder import VoiceEncoder
from models.face_decoder import FaceDecoder
import torch
from matplotlib import pyplot as plt
import librosa
import data_preprocessing.audio_spectrograms as audio_processing
import numpy as np
import torch.nn as nn
from tqdm import tqdm


ACCEPTED_AUDIO_EXTENSIONS = ['.m4a', '.wav']


parser = ArgumentParser("Use trained VoiceEncoder and FaceDecoder to generate face image \
                         out of audio file for every adio in the given directory")

parser.add_argument("--test-set-path", type=str, required=True,
                    help="Path to the test set of a dataset")

parser.add_argument("--save-path", type=str, required=True,
                    help="Path to saved images")

parser.add_argument("--ve-path", type=str, required=True,
                    help="Path to the saved trained model of AST model")

parser.add_argument("--face-decoder-path", type=str, required=True,
                    help="Path to the saved trained model of face decoder")

parser.add_argument("--gpu", type=int, default=0,
                    help="-1 for cpu prediction")

parser.add_argument("--audio-length", default=6, type=float,
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


def get_device(choice):
    if choice >= 0:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{choice}")
        else:
            raise ValueError("GPU training was chosen but cuda is not available")
    else:
        device = torch.device("cpu")
    return device


def main():
    args = parser.parse_args()
    
    device = get_device(args.gpu)

    face_decoder = FaceDecoder().to(device)
    face_decoder.eval()
    face_decoder_checkpoint = torch.load(args.face_decoder_path)
    face_decoder.load_state_dict(face_decoder_checkpoint["model_state_dict"])

    voice_encoder = VoiceEncoder().to(device)
    voice_encoder.eval()
    voice_encoder.eval()

    voice_encoder_checkpoint = torch.load(args.ve_path)
    voice_encoder.load_state_dict(voice_encoder_checkpoint["model_state_dict"])

    for root, _, files in tqdm(os.walk(args.test_set_path), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            file_path = os.path.join(root, file_name)
            file_base, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_AUDIO_EXTENSIONS:
                continue
            
            normalized_file_path = audio_processing.normalize_audio(file_path, root, file_base)
            waveform, _ = librosa.load(normalized_file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
            waveform = audio_processing.stretch_audio(args, waveform)
            spectrogram = audio_processing.compute_spectrograms(args, waveform)
            spectrogram = audio_processing.power_law_compression(args, spectrogram)
            spectrogram = torch.tensor(spectrogram).unsqueeze(0).to(device)
            os.remove(normalized_file_path)
            
            with torch.no_grad():
                voice_encoder_embeddings = voice_encoder(spectrogram)
                landmarks_predicted, images_predicted = face_decoder(voice_encoder_embeddings)

            _, axes = plt.subplots(1, 1, figsize=(12, 8))
            temp_image = (images_predicted[0].cpu() * 255).to(torch.uint8)
            axes.imshow(temp_image.permute(1, 2, 0))
            axes.axis("off")

            os.makedirs(f"{args.save_path}/{os.path.basename(root)}", exist_ok=True)
            plt.savefig(f"{args.save_path}/{os.path.basename(root)}/predicted_{os.path.basename(root)}.jpg", bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.close()
            break



if __name__ == "__main__":
    main()
