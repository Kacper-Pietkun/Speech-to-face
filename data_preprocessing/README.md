# Audio preprocessing

The `audio.py` script can be used to convert raw audio files (with extensions such as: ".wav", ".m4a", etc.) into spectrograms, that can be directly used as input to the voice encoder neural network.
Spectrogram properties can be modified by changing script's input arguments. The default values are set to be the same as in [Speech2Face: Learning the Face Behind a Voice](https://arxiv.org/abs/1905.09773) article.
You need to specify `--data-dir` and `--save-dir` arguments which tell where the data is located and where it should be saved. Script is adjusted to traverse nested directories, so you don't have to put all of the audio files in the same directory. The script will recreate the nested structure of the directories in `--save-dir` directory.

## Example usage
```shell
python audio.py --data-dir <path_to_the_directory_where_data_is_stored> --save-dir <path_to_the_directory_where_spectrograms_will_be_saved>
```