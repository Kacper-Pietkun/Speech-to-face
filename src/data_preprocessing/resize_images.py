import os
from PIL import ImageOps
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import face_recognition
import torch
import torchvision.transforms as transforms


ACCEPTED_IMAGE_EXTENSIONS = ['.jpg']

parser = ArgumentParser(description="Resizing images to a sizes required by FaceDecoder (224x224)")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the directory where image files are located")

parser.add_argument("--save-dir", required=True, type=str,
                    help="Absolute path to the directory where face landmarks will be saved")


def load_image(file_path):
    image = Image.open(str(file_path))
    image = ImageOps.exif_transpose(image)
    return image


def save_resized_image(args, root, file_name, resized_image):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    new_file_path = os.path.join(save_dir, file_name)
    resized_image.save(new_file_path)


def resize_image(image, transform):
    image = transform(image)
    image = image.permute(1, 2, 0)
    resized_image = image * 255
    resized_image = resized_image.to(torch.uint8)
    resized_image = np.array(resized_image)
    resized_image = Image.fromarray(resized_image)
    return resized_image


def main():
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    for root, _, files in tqdm(os.walk(args.data_dir), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            _, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_IMAGE_EXTENSIONS:
                continue
            file_path = os.path.join(root, file_name)
            image = load_image(file_path)
            resized_image = resize_image(image, transform)
            save_resized_image(args, root, file_name, resized_image)


if __name__ == "__main__":
    main()
