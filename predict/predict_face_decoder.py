import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from argparse import ArgumentParser
from src.models.face_decoder import FaceDecoder
import torch
from matplotlib import pyplot as plt
import src.data_preprocessing.image_face_embeddings as embedding_processing
import src.data_preprocessing.image_face_landmarks as landmarks_processing
import torchvision.transforms as transforms
import numpy as np


parser = ArgumentParser("Use trained FaceDecoder to generate face image out of image's embedding")

parser.add_argument("--image-file-path", type=str, required=True,
                    help="Path to the image")

parser.add_argument("--face-decoder-path", type=str, required=True,
                    help="Path to the saved trained model of face decoder")

parser.add_argument("--face-encoder-model", default="vgg_face_serengil", type=str, choices=["vgg_face_serengil", "vgg_face_16_rcmalli"],
                    help="Backend for calculating embedding (features) of images")

parser.add_argument("--face-encoder-weights-path", required=True, type=str,
                    help="Absolute path to a file where model weights for face encoder backend are stored")

parser.add_argument("--gpu", type=int, default=0,
                    help="-1 for cpu prediction")


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

    face_encoder = embedding_processing.load_face_encoder(args.face_encoder_model,
                                                         args.face_encoder_weights_path).to(device)
    face_encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    original_image = embedding_processing.load_image(args.image_file_path)
    original_landmarks = landmarks_processing.calculate_landmarks(np.array(original_image))
    img_tensor = transform(original_image).to(device)
    embedding = embedding_processing.calculate_embedding(face_encoder, img_tensor)
    embedding = torch.tensor(embedding).unsqueeze(0).to(device)

    with torch.no_grad():
        landmarks_predicted, images_predicted = face_decoder(embedding)

    _, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    temp_landmarks = original_landmarks.cpu().view(72, 2)
    x, y =zip(*temp_landmarks.squeeze(0))
    axes[0].scatter(x, y, c='red', marker='o', s=2)

    temp_image = (images_predicted[0].cpu() * 255).to(torch.uint8)
    axes[1].imshow(temp_image.permute(1, 2, 0))
    axes[1].set_title("Predicted image")
    axes[1].axis("off")
    temp_landmarks = landmarks_predicted[0].cpu().view(72, 2)
    x, y =zip(*temp_landmarks.squeeze(0))
    axes[1].scatter(x, y, c='red', marker='o', s=2)

    plt.show()


if __name__ == "__main__":
    main()
