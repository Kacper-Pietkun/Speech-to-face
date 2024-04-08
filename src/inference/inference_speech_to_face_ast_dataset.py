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
import torch.nn as nn
from tqdm import tqdm


ACCEPTED_AUDIO_EXTENSIONS = ['.m4a', '.wav']


parser = ArgumentParser("Use trained AST and FaceDecoder to generate face image \
                         out of audio file for every adio in the given directory")

parser.add_argument("--test-set-path", type=str, required=True,
                    help="Path to the test set of a dataset")

parser.add_argument("--save-path", type=str, required=True,
                    help="Path to saved images")

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

parser.add_argument("--mean", default=-5.460994, type=float, #-1.3173023e-06
                    help="Mean value of the audio files (calculated on the training set)")

parser.add_argument("--std", default=3.1129124, type=float, #0.039394222
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
    head = ast.classifier
    new_head = nn.Sequential(
        head,
        nn.ReLU()
    )
    ast.classifier = new_head
    ast.eval()

    ast_checkpoint = torch.load(args.ast_path)
    ast.load_state_dict(ast_checkpoint["model_state_dict"])

    feature_extractor = ASTFeatureExtractor(sampling_rate=args.sampling_rate, mean=args.mean, std=args.std)

    ctr = 1
    for root, _, files in tqdm(os.walk(args.test_set_path), desc="Outer Loop"):
        print(os.path.basename(root))
        if os.path.basename(root) != "Fred_Rutten" and \
            os.path.basename(root) != "Mancini" and os.path.basename(root) != "Jacques_Gamblin":
            continue
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            file_path = os.path.join(root, file_name)
            file_base, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_AUDIO_EXTENSIONS:
                continue
            
            try:
                normalized_file_path = audio_processing.normalize_audio(file_path, root, file_base)
                waveform, _ = librosa.load(normalized_file_path, duration=args.audio_length, sr=args.sampling_rate, mono=True)
                waveform = audio_processing.stretch_audio(args, waveform)
            except:
                continue

            inputs = feature_extractor(waveform, sampling_rate=args.sampling_rate, padding="max_length", return_tensors="np")
            input_values = torch.tensor(inputs.input_values).to(device)
            os.remove(normalized_file_path)
            
            with torch.no_grad():
                voice_encoder_embeddings = ast(input_values)
                voice_encoder_embeddings = voice_encoder_embeddings.logits
                landmarks_predicted, images_predicted = face_decoder(voice_encoder_embeddings)

            _, axes = plt.subplots(1, 1)
            temp_image = (images_predicted[0].cpu() * 255).to(torch.uint8)
            axes.imshow(temp_image.permute(1, 2, 0))
            axes.axis("off")
            os.makedirs(f"{args.save_path}/{os.path.basename(root)}", exist_ok=True)
            plt.savefig(f"{args.save_path}/{os.path.basename(root)}/predicted_{os.path.basename(root)}_{ctr}.jpg", bbox_inches='tight', pad_inches=0)
            ctr += 1
            plt.clf()
            plt.close()




if __name__ == "__main__":
    main()
