from argparse import ArgumentParser
from PIL import ImageOps
from PIL import Image
import numpy as np
import os

ACCEPTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

parser = ArgumentParser(description="Getting features out of images of faces")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the directory where audio files are located")

parser.add_argument("--save-dir", required=True, type=str,
                    help="Absolute path to the directory where spectrograms will be saved")

parser.add_argument("backend", default="vgg_face", type=str, choices=["vgg_face"],
                    help="Backend for calculating embedding (features) of images")


def load_image(file_path):
    image = Image.open(str(file_path))
    image = ImageOps.exif_transpose(image)
    return image
        

def calculate_embedding(args, image):
    ...


def save_embedding(args, root, file_name, spectrogram):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    name, _ = os.path.splitext(file_name)
    name += ".npy"
    new_file_path = os.path.join(save_dir, name)
    np.save(new_file_path, spectrogram)


def main():
    args = parser.parse_args()

    for root, _, files in os.walk(args.data_dir):
        for file_name in files:
            _, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_IMAGE_EXTENSIONS:
                continue
            file_path = os.path.join(root, file_name)
            image = load_image(file_path)
            embedding = calculate_embedding(args, image)
            save_embedding(args, embedding)


if __name__ == "__main__":
    main()
