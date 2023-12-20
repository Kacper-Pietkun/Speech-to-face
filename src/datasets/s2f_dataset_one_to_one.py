import os
from torch.utils.data import Dataset
import numpy as np


class S2fDatasetOneToOne(Dataset):
    def __init__(self, root_folder, is_ast=False, transform=None, target_transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.target_transform = target_transform
        self.emebddings_paths = []
        self.spectrograms_paths = []
        self.find_paths()
        self.individuals_number = len(self.emebddings_paths)
        self.is_ast = is_ast
        assert len(self.spectrograms_paths) == len(self.emebddings_paths), "Number of spectrograms is not equal to number of embeddings"

    def find_paths(self):
        for person_dir in os.listdir(self.root_folder):
            spectrogram_path = os.path.join(self.root_folder, person_dir, "audios")
            person_spectrograms = self.get_list_of_data_paths(spectrogram_path)
            embedding_path = os.path.join(self.root_folder, person_dir, "images")
            person_embeddings = self.get_list_of_data_paths(embedding_path)

            num_spec = len(person_spectrograms)
            num_emb = len(person_embeddings)
            if num_spec == 0 or num_emb == 0:
                continue # skip identities which does not have spectrograms or embeddings
            
            min_number = num_spec if num_spec < num_emb else num_emb
            self.spectrograms_paths += person_spectrograms[:min_number]
            self.emebddings_paths += person_embeddings[:min_number]
            assert len(self.emebddings_paths) == len(self.spectrograms_paths), "Unexpected number of spectrograms or embeddings (sizes are not equal)"

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
        return len(self.emebddings_paths)

    def __getitem__(self, idx):
        spectrogram_path = self.spectrograms_paths[idx]
        embedding_path = self.emebddings_paths[idx]
        spectrogram = np.load(spectrogram_path)
        embedding = np.load(embedding_path)
        if self.is_ast:
            spectrogram = spectrogram.squeeze()
        return spectrogram, embedding
