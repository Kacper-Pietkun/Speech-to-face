# Audio preprocessing

The `audio.py` script can be used to convert raw audio files (with extensions such as: ".wav", ".m4a", etc.) into spectrograms, that can be directly used as input to the voice encoder neural network.
Spectrogram properties can be modified by changing script's input arguments. The default values are set to be the same as in [Speech2Face: Learning the Face Behind a Voice](https://arxiv.org/abs/1905.09773) article.
You need to specify `--data-dir` and `--save-dir` arguments which tell where the data is located and where it should be saved. Script is adjusted to traverse nested directories, so you don't have to put all of the audio files in the same directory. The script will recreate the nested structure of the directories in `--save-dir` directory.

## Usage
```shell
python audio.py --data-dir <path_to_the_directory_where_data_is_stored> --save-dir <path_to_the_directory_where_spectrograms_will_be_saved>
```

# Images preprocessing

The `image.py` script can be used to calculate embeddings for images of faces. Those embeddings are feature representation of faces (Voice encoder model will try to learn to generate such embeddings, basing on speech).
You need to specify `--data-dir` and `--save-dir` arguments which tell where the data is located and where it should be saved. Script is adjusted to traverse nested directories, so you don't have to put all of the image files in the same directory. The script will recreate the nested structure of the directories in `--save-dir` directory.
You can choose backend i.e. model that will be used to generate embeddings. Currently, we support two backends:
- VGG-face model from https://github.com/serengil/deepface
- VGG-face (16) model from https://github.com/rcmalli/keras-vggface
You can choose them with `--backend` parameter. You also have to provide a path to the weights for the chosen model. We downloaded them from repositories listed above, and after converting them from Tensorflow to PyTorch, we saved them in the following directory `src/models/pytorch_weights`.

## Usage
```shell
python image.py --data-dir <path_to_the_directory_where_data_is_stored> --save-dir <path_to_the_directory_where_embeddings_will_be_saved> --backend-weights <path_to_the_file_where_chosen_backend_weights_are_saved> --backend <name_of_the_chosen_backend>
```
