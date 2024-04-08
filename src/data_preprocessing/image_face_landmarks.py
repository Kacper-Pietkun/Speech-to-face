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

parser = ArgumentParser(description="Getting face landmarks out of images of faces")

parser.add_argument("--data-dir", required=True, type=str,
                    help="Absolute path to the directory where image files are located")

parser.add_argument("--save-dir", required=True, type=str,
                    help="Absolute path to the directory where face landmarks will be saved")


def load_image(file_path):
    image = Image.open(str(file_path))
    image = ImageOps.exif_transpose(image)
    return image


def save_face_landmarks(args, root, file_name, face_landmarks):
    additional_dirs = os.path.relpath(root, start=args.data_dir)
    save_dir = os.path.join(args.save_dir, additional_dirs)
    os.makedirs(save_dir,  exist_ok=True)
    name, _ = os.path.splitext(file_name)
    name += "_landmarks.npy"
    new_file_path = os.path.join(save_dir, name)
    np.save(new_file_path, face_landmarks)


def calculate_landmarks(image):
    landmarks = face_recognition.face_landmarks(image)
    landmarks_tensor = torch.empty(0)
    if not landmarks:
        return None
    for key in landmarks[0].keys():
        landmarks_tensor = torch.cat((landmarks_tensor, torch.tensor(landmarks[0][key])))
    landmarks_tensor = landmarks_tensor.ravel()
    return landmarks_tensor


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
            img_tensor = transform(image).permute(1, 2, 0)
            img_tensor = img_tensor * 255
            img_tensor = img_tensor.to(torch.int)
            img = np.array(img_tensor, dtype=np.uint8)
            face_landmarks = calculate_landmarks(img)
            if face_landmarks is not None:
                save_face_landmarks(args, root, file_name, face_landmarks)
            else:
                print(f"Landmarsk not saved for: {file_path}")


if __name__ == "__main__":
    main()
