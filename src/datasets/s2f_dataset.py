import os
from torch.utils.data import Dataset
import numpy as np


class S2fDataset(Dataset):
    def __init__(self, root_folder, transform=None, target_transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.target_transform = target_transform
        self.emebddings_paths = []
        self.spectrograms_paths = []
        self.find_paths()
        self.individuals_number = len(self.emebddings_paths)

    def find_paths(self):
        for person_dir in os.listdir(self.root_folder):
            spectrogram_path = os.path.join(self.root_folder, person_dir, "audios")
            self.spectrograms_paths.append(self.get_list_of_data_paths(spectrogram_path))

            embedding_path = os.path.join(self.root_folder, person_dir, "images")
            self.emebddings_paths.append(self.get_list_of_data_paths(embedding_path))

    def get_list_of_data_paths(self, path):
        data_paths = []
        for root, _, files in os.walk(path):
            for file_name in files:
                _, extension = os.path.splitext(file_name)
                if extension != '.npy':
                    continue
                file_path = os.path.join(root, file_name)
                data_paths.append(file_path)
        return data_paths

    def __len__(self):
        length = 0
        for i in range(self.individuals_number):
            length += len(self.emebddings_paths[i]) * len(self.spectrograms_paths[i])
        return length

    def __getitem__(self, idx):
        spectrogram_idx = -1
        embedding_idx = -1
        person_idx = -1
        start_idx = 0
        for i in range(self.individuals_number):
            num_elems = len(self.emebddings_paths[i]) * len(self.spectrograms_paths[i])
            if idx >= start_idx and idx < start_idx + num_elems:
                idx -= start_idx
                spectrogram_idx = idx // len(self.emebddings_paths[i])
                embedding_idx = idx % len(self.emebddings_paths[i])
                person_idx = i
                break
            start_idx += num_elems
        spectrogram_path = self.spectrograms_paths[person_idx][spectrogram_idx]
        embedding_path = self.emebddings_paths[person_idx][embedding_idx]
        spectrogram = np.load(spectrogram_path)
        embedding = np.load(embedding_path)
        return spectrogram, embedding
