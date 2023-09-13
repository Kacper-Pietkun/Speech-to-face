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


Except for the folders named `audios` and `images`, names of the folders and files are arbitrary. Spectrograms and face embeddings inside `audios` and `images` directories, can but don't need to be grouped in subfolders. Note that number of spectrograms in `audios` directory doesn't need to be the same as the number of face embeddings in `images` directory, for a given person.

You can generate spectrograms and face embeddings using `audio_spectrograms.py` and  `image_face_embeddings.py` scripts in `src\data_preprocessing` folder in this repository.


# FaceDecoderDataset
Custom dataset used for training face decoder. It requires only one argument - path to the root folder of the dataset.

Dataset structure should be created as follows:
- dataset_name
    - Person_1
        - face images in .jpg format
        - face embeddings in .npy format
        - face landmarks in .npy format
    - Person_2
        - face images in .jpg format
        - face embeddings in .npy format
        - face landmarks in .npy format
    - ...
        - ...
    - Person_n
        - face images in .jpg format
        - face embeddings in .npy format
        - face landmarks in .npy format


Names of the folders and files are arbitrary, except for two things:
- face embedding files must have the same name as face image file, but with `_embedding` suffix
- face landmark files must have the same name as face image file, but with `_landmarks` suffix

For example, when having an face image file with name `my_face.jpg`, then the embedding file must be named `my_face_embedding.npy` and the landmarks file must be named `my_face_landmarks.npy`

Note that each person can have a different number of (image, embedding, landmarks) files. Additionally, for each face image file, there must be equivalent embedding and landmarks file, otherwise it won't be used during training face decoder. You can generate face embeddings and face landmarks from a face image using `image_face_embeddings.py` and  `image_face_landmarks.py` scripts in `src\data_preprocessing` folder in this repository.