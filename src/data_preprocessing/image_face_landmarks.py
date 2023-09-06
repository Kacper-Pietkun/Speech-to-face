import os
from argparse import ArgumentParser
from PIL import ImageOps
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import face_recognition


ACCEPTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

parser = ArgumentParser(description="Getting face landmarks out of images of faces")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the directory where image files are located")

parser.add_argument("--save-dir", required=True, type=str,
                    help="Absolute path to the directory where face landmarks will be saved")


def save_face_landmarks(args, root, file_name, face_landmarks):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    name, _ = os.path.splitext(file_name)
    name += "_landmarks.npy"
    new_file_path = os.path.join(save_dir, name)
    np.save(new_file_path, face_landmarks)


def main():
    args = parser.parse_args()

    for root, _, files in tqdm(os.walk(args.data_dir), desc="Outer Loop"):
        for file_name in tqdm(files, desc="Inner Loop", leave=False):
            _, extension = os.path.splitext(file_name)
            if extension not in ACCEPTED_IMAGE_EXTENSIONS:
                continue
            file_path = os.path.join(root, file_name)
            image = face_recognition.load_image_file(file_path)
            face_landmarks = face_recognition.face_landmarks(image)
            save_face_landmarks(args, root, file_name, face_landmarks)


if __name__ == "__main__":
    main()
