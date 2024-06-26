# Audio preprocessing - spectrograms (+ audio normalization)

The `audio_spectrograms.py` script can be used to convert raw audio files (with extensions such as: ".wav", ".m4a", etc.) into spectrograms, that can be directly used as input to the VE_conv (VoiceEncoder architecture from Speech-to-Face paper).
Spectrogram properties can be modified by changing script's input arguments. The default values are set to be the same as in [Speech2Face: Learning the Face Behind a Voice](https://arxiv.org/abs/1905.09773) article. Additionally audio files are being normalized before computing spectrograms. Normalization consists of following modifications: converting file to ".wav" format with PCM16 codec, setting sample rate to 16KHz, saving file as one channel, normalization with RMS algorithm with target size equal to -23.
You need to specify `--data-dir` and `--save-dir` arguments which tell where the audio files are located and where it should save spectrograms. Script is adjusted to traverse nested directories, so you don't have to put all of the audio files in the same directory. The script will recreate the nested structure of the directories in `--save-dir` directory. Saved spectrograms will have the same names as the original audio files, but they will be saved with `.npy` extension.

## Usage
```shell
python audio_spectrograms.py --data-dir <path_to_the_directory_where_data_is_stored> \
                             --save-dir <path_to_the_directory_where_spectrograms_will_be_saved>
```

The `ast_audio_spectrogram.py` works in a similar way. It can be used to preprocess raw audio files to be compatible with AST (Audio Spectrogram Transformer) network. Usage of this script is also similar, however you need to pass two additional arguments: `mean` and `std`, which represent values that will be used to normalize log-Mel features. You can use the default ones (the one proposed by the authors), however, you can calculate them based on your dataset with `get_audio_norm_stats.py` script.

# Images preprocessing - face embeddings

The `image_face_embeddings.py` script can be used to calculate embeddings for images of faces. Those embeddings are feature representation of faces (Voice encoder model will try to learn to generate such embeddings, basing on speech).
You need to specify `--data-dir` and `--save-dir` arguments which tell where the image files are located and where it should save face embeddings. Script is adjusted to traverse nested directories, so you don't have to put all of the image files in the same directory. The script will recreate the nested structure of the directories in `--save-dir` directory. Saved embeddings will have the same names as the original image files with `_embedding` suffix, and they will be saved with `.npy` extension.
You can choose backend i.e. model that will be used to generate embeddings. Currently, we support two backends:
- VGG-face model from https://github.com/serengil/deepface
- VGG-face (16) model from https://github.com/rcmalli/keras-vggface

You can choose them with `--backend` parameter. You also have to provide a path to the weights for the chosen model. You can download weights for those models from the repositories listed above, however they will not work right away, because they are in TensorFlow format and in our project we use PyTorch. To convert the weights of the model by yourself go to the `src/tensorflow_to_pytorch` directory and follow the steps listed there (We decided not to place the converted weights here in the repo, because they are too big. If you need any help, we can place the weights on a google drive).

## Usage
```shell
python image_face_embeddings.py --data-dir <path_to_the_directory_where_data_is_stored> \
                                --save-dir <path_to_the_directory_where_embeddings_will_be_saved> \
                                --backend-weights <path_to_the_file_where_chosen_backend_weights_are_saved> \
                                --backend <name_of_the_chosen_backend>
```


# Images preprocessing - face landmarks

The `image_face_landmarks.py` script can be used to calculate face landmarks for images of faces. Script uses face_recognition package in order to do that. You need to specify `--data-dir` and `--save-dir` arguments which tell where the image files are located and where it should save face landmarks. Script is adjusted to traverse nested directories, so you don't have to put all of the image files in the same directory. The script will recreate the nested structure of the directories in `--save-dir` directory. Saved landmarks will have the same names as the original images files with `_landmark` suffix, and `.npy` extension. Note that landmarks returned by face_recognition's function `face_landmarks` are converted to a one dimensional torch tensor with shape (144). (Additioanlly, be aware that the face_recognition package is not alway able to detect the landmarks of all of the faces).

## Usage
```shell
python image_face_landmarks.py --data-dir <path_to_the_directory_where_data_is_stored> \
                               --save-dir <path_to_the_directory_where_landmarks_will_be_saved>
```


# Images preprocessing - resizing image

The `resize_images.py` script can be used to resize images to a size required by FaceDecoder model (224x224). Resizing images is optional, because every script checks whether images have appropriate sizes and resize them if necessary (scripts like FaceDecoder training, calculating image embeddings and image landmarks). Script is adjusted to traverse nested directories, so you don't have to put all of the image files in the same directory. The script will recreate the nested structure of the directories in `--save-dir` directory. Resized images will have the same name and extension as the original images.

## Usage
```shell
# --data-dir can be equal to --save-dir, in that case original images will be overwritten
python resize_images.py --data-dir <path_to_the_directory_where_data_is_stored> \
                        --save-dir <path_to_the_directory_where_resized_images_will_be_saved>
```


# Images preprocessing - normalizing directory names
Before splitting dataset to train/val/test datasets it is required to normalize directory names (names of the people) so that they contain only ASCII letters. It is required because deepface package, which we use for face analysis, cannot process non ASCII letters directories. (We use deepface to analyze images in order to split dataset into train/val/test sets in stratified way (same ratio of age_group, gender, race))

## Usage
```shell
python normalize_directory_names.py --data-dir <path_to_the_root_folder_of_the_dataset>
```