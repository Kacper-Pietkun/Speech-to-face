import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FaceDecoderDataset(Dataset):
    def __init__(self, root_folder, transform=None, target_transform=None):
        self.root_folder = root_folder
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform
        self.data_paths = self.find_paths()
        self.individuals_number = len(self.data_paths)

    def find_paths(self):
        paths = []
        for person_dir in os.listdir(self.root_folder):
            person_list = []
            person_path = os.path.join(self.root_folder, person_dir)
            for root, _, files in os.walk(person_path):
                for file_name in files:
                    base, extension = os.path.splitext(file_name)
                    # For each image in dataset, there must be embedding and landmarks file
                    if extension != '.jpg' or \
                        not os.path.isfile(os.path.join(root, f"{base}_landmarks.npy")) or \
                        not os.path.isfile(os.path.join(root, f"{base}_embedding.npy")):
                        continue
                    image_path = os.path.join(root, base)
                    person_list.append(image_path)
            paths.append(person_list)
        return paths
    
    def __len__(self):
        length = 0
        for i in range(self.individuals_number):
            length += len(self.data_paths[i])
        return length
    
    def __getitem__(self, idx):
        person_idx = -1
        data_idx = -1
        start_idx = 0
        for i in range(self.individuals_number):
            num_elems = len(self.data_paths[i])
            if idx >= start_idx and idx < start_idx + num_elems:
                idx -= start_idx
                data_idx = idx
                person_idx = i
                break
            start_idx += num_elems
        image_path = f"{self.data_paths[person_idx][data_idx]}.jpg"
        embedding_path = f"{self.data_paths[person_idx][data_idx]}_embedding.npy"
        landmarks_path = f"{self.data_paths[person_idx][data_idx]}_landmarks.npy"
        image = Image.open(str(image_path))
        image = ImageOps.exif_transpose(image)
        image = self.transform(image)
        embedding = np.load(embedding_path)
        landmarks = np.load(landmarks_path)
        return image, embedding, landmarks
