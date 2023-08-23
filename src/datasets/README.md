# S2fDataset
Custom dataset used for training voice encoder. It requires only one argument - path to the root folder of the dataset.

Dataset structure should be created as follows:
- dataset_name
    - Person_1
        - audios
            - spectrograms in .npy format
        - images
            - face embeddings in .npy format
    - Person_2
        - audios
            - spectrograms in .npy format
        - images
            - face embeddings in .npy format
    - ...
        - ...
        - ...
    - Person_n
        - audios
            - spectrograms in .npy format
        - images
            - face embeddings in .npy format


Except for the folders named `audios` and `images`, names of the folders and files are arbitrary. Spectrograms and face embeddings inside `audios` and `images` directories, can but don't need to be grouped in subfolders.
You can convert spectrograms and face embeddings to `.npy` format using `audio.py` and  `image.py` scripts in `data_preprocessing` folder in this repository.
