# S2fDataset
Custom dataset used for training voice encoder. It requires only one argument - path to the root folder of the dataset.

There are three approaches for a dataset that we implemented. `S2fDatasetAlltoAll` class will create a training pair (audio spectrogram, face embedding) for every possibility for every person. it means that each spectrogram will be matched with each face embedding. So for eaxample, when a person has have 4 spectrograms and 7 embeddings, then this person will have 4*7=28 training pairs. Another approach is implemented in `S2fDatasetOneToOne` class, where each spectrogram is matched with one face embedding. It means that surplus files will be rejected. For example, when a person has have 4 spectrograms and 7 embeddings, then this person will have only 4 training pairs (additional 3 embeddings will be rejected). Lastly, thre is `S2fDatasetAlltoOne`, which will match every audio clip with one, the same face embedding (you can choose by specifing an `n` argument, to match only `n` audio clips, so it behaves more like n-to-one dataloader).

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

## Splitting FaceDecoder dataset into train/validation/test sets

Use `face_decoder_split.py` script to split FaceDecoder dataset into train/validation/test sets. To ensure that splitted dataset is balanced we are using stratified split. Firstly we determine three categories for each person using deepface library - race, age_group and gender (You can determine which categories will be calculated passing `gender`, `race` and `age` arguments to the script). This step is required, because if we splitted dataset randomly, then there would be chance that validation and test sets consist only of woman images and that would skew the results. Because of this stratifying step, process of dataset splitting takes a lot of time.

### Usage
```shell
# Example for stratified split for people's age and race
python face_decoder_split.py --dataset-path <path_to_the_root_folder_of_the_dataset> \
                             --save-path <path where train/validation/test folders will be created> \
                             --train-size <size of the train set <sizes of validation and test sets are set to (1-train_size)/2)> \
                             --age \
                             --race
```