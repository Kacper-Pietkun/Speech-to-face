import os
import sys
from argparse import ArgumentParser
from PIL import ImageOps
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.face_encoder import VGGFace16_rcmalli, VGGFace_serengil
import numpy as np
import os

ACCEPTED_IMAGE_EXTENSIONS = ['.jpg']

parser = ArgumentParser(description="Getting features out of images of faces")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the directory where image files are located")

parser.add_argument("--save-dir", required=True, type=str,
                    help="Absolute path to the directory where face embeddings will be saved")

parser.add_argument("--backend", default="vgg_face_serengil", type=str, choices=["vgg_face_serengil", "vgg_face_16_rcmalli"],
                    help="Backend for calculating embedding (features) of images")

parser.add_argument("--backend-weights-path", required=True, type=str,
                    help="Absolute path to a file where model weights for face encoder backend are stored")


def load_image(file_path):
    image = Image.open(str(file_path))
    image = ImageOps.exif_transpose(image)
    return image


def calculate_embedding(model, img_tensor):
    img_tensor = img_tensor.unsqueeze(0)
    embedding = model(img_tensor, get_embedding=True)
    embedding = embedding.squeeze()
    embedding = embedding.detach()
    return embedding


def load_face_encoder(backend, weights_path):
    if backend == "vgg_face_serengil":
        model = VGGFace_serengil()
    elif backend == "vgg_face_16_rcmalli":
        model = VGGFace16_rcmalli()
    else:
        raise ValueError(f"Unrecognized backend for face encoder: {backend}")
    
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def save_embedding(args, root, file_name, embedding):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    name, _ = os.path.splitext(file_name)
    name += "_embedding.npy"
    new_file_path = os.path.join(save_dir, name)
    np.save(new_file_path, embedding)


def main():
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    model = load_face_encoder(args.backend, args.backend_weights_path)

    for root, _, files in tqdm(os.walk(args.data_dir), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            _, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_IMAGE_EXTENSIONS:
                continue
            file_path = os.path.join(root, file_name)
            image = load_image(file_path)
            img_tensor = transform(image)
            embedding = calculate_embedding(model, img_tensor)
            save_embedding(args, root, file_name, embedding)


if __name__ == "__main__":
    main()
