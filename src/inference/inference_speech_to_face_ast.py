import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from argparse import ArgumentParser
from transformers import AutoModelForAudioClassification, ASTFeatureExtractor
from models.face_decoder import FaceDecoder
import torch
from matplotlib import pyplot as plt
import librosa
import data_preprocessing.ast_audio_preprocess as audio_processing
import numpy as np


parser = ArgumentParser("Use trained VoiceEncoder and FaceDecoder to generate face image out of audio file")

parser.add_argument("--audio-file-path", type=str, required=True,
                    help="Path to the audio file")

parser.add_argument("--ast-path", type=str, required=True,
                    help="Path to the saved trained model of AST model")

parser.add_argument("--face-decoder-path", type=str, required=True,
                    help="Path to the saved trained model of face decoder")

parser.add_argument("--gpu", type=int, default=0,
                    help="-1 for cpu prediction")

parser.add_argument("--audio-length", default=10.26, type=float,
                    help="length of the audio that will be converted to spectogram. Longer audio \
                          files will be clipped, and shorter files will be repeated")

parser.add_argument("--sampling-rate", default=16000, type=int,
                    help="The number of samples per second of audio")

parser.add_argument("--mean", default=-4.2677393, type=int, #-1.3173023e-06
                    help="Mean value of the audio files (calculated on the training set)")

parser.add_argument("--std", default=4.5689974, type=int, #0.039394222
                    help="Standard deviation value of the audio files (calculated on the training set)")


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

    ast = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=4096, ignore_mismatched_sizes=True).to(device)
    ast.eval()

    ast_checkpoint = torch.load(args.ast_path)
    ast.load_state_dict(ast_checkpoint["model_state_dict"])

    feature_extractor = ASTFeatureExtractor(sampling_rate=args.sampling_rate, mean=args.mean, std=args.std)
    root = os.path.dirname(args.audio_file_path)
    file_base, _ = os.path.splitext(os.path.basename(args.audio_file_path))
    normalized_file_path = audio_processing.normalize_audio(args.audio_file_path, root, file_base)
    waveform, _ = librosa.load(args.audio_file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
    waveform = audio_processing.stretch_audio(args, waveform)

    inputs1 = feature_extractor(waveform, sampling_rate=args.sampling_rate, padding="max_length", return_tensors="np")
    input_values = torch.tensor(inputs1.input_values).to(device)
    os.remove(normalized_file_path)




    
    with torch.no_grad():
        voice_encoder_embeddings = ast(input_values)
        voice_encoder_embeddings = voice_encoder_embeddings.logits
        landmarks_predicted, images_predicted = face_decoder(voice_encoder_embeddings)

    _, axes = plt.subplots(1, 1, figsize=(12, 8))
    temp_image = (images_predicted[0].cpu() * 255).to(torch.uint8)
    axes.imshow(temp_image.permute(1, 2, 0))
    axes.set_title("Reconstructed image")
    axes.axis("off")
    temp_image = landmarks_predicted[0].cpu().view(72, 2)
    x, y =zip(*temp_image.squeeze(0))
    axes.scatter(x, y, c='red', marker='o', s=2)
    plt.show()


if __name__ == "__main__":
    main()
