# Face Encoder
We don't train FaceEncoder since we are using already trained model

# Voice Encoder
Run `train_ve_conv.py` script to train VE_conv VoiceEncoder

Loss functions conists of three components:
    1. Distance between ground truth 4096-D face feature vector and the predicted 4096-D face feature vector
    2. Difference between the activation of the last layer of the pre-trained face encoder (thus you need to specify which model to use with `--face-encoder` parameter)
    3. Difference between the activation of the first layer of the pre-trained face decoder

You need to specify following parameters:
- Path to a dataset - train and validation sets paths separately with `--train-dataset-path` and `--val-dataset-path`
- Because of this loss function you need to choose FaceEncoder model type with `--face-encoder` parameter and you need to specify path to the trained weights of that model with `--face-encoder-weights-path` parameter
- Path to the trained weights of the FaceDecoder with `--face-decoder-weights-path` parameter
- Path to where all the results will be saved. The results are - trained weights of VoiceEncoder and history training. To do that use `--save-folder-path` parameter.

Additionally:
- You can stop training whenever you want and rerun the script to continue the training (you need to specify `--continue-training-path` paramater, which tells the programe where the trained weights of the VoiceEncoder are located at).
- If possible train VoiceEncoder with GPU, by setting `--gpu` parameter to 1, because CPU training is very slow


Run `train_ast.py` if you want to train AST VoiceEncoder. The script is almost identical to the `train_voice_encoder.py` script, however there are some differences. For example, you can use early stopping (specify `--early-stopping` parameter). Moreover, the AST model training is splitted into two parts. During the first part, whole model is frozen except the head, which is trained. During the second part whole model is unfrozen and the model is fine-tuned.

# Face Decoder

Run `train_face_decoder.py` script to train FaceDecoder

Loss functions conists of two components (the third component was not implemented):
    1. Difference between ground truth and predicted face landmarks (MSE) 
    2. Difference between ground thruth and predicted textures (MAE)
    3. Third component which is described in paper that penalizes the modeel based on a difference between ground truth and generated embeddings in not implemented in our project, because we didn't implement the differentialbe warping

You need to specify following parameters:
- Path to a dataset - train and validation sets paths separately with `--train-dataset-path` and `--val-dataset-path`
- Path to where all the results will be saved. The results are - trained weights of FaceDecoder and history training. To do that use `--save-folder-path` parameter.

Additionally:
- By passing `--save-images` parameter to the script, at the beginning of each epoch script will save an exemplary generated face from the train and validation sets (so you can see how good the model is). The generated faces will be plotted next to the original faces from the train and validation sets.
- You can stop training whenever you want and rerun the script to continue the training (you need to specify `--continue-training-path` paramater, which tells the programe where the trained weights of the VoiceEncoder are located at).
- If possible train VoiceEncoder with GPU, by setting `--gpu` parameter to 1, because CPU training is very slow

Note:
- In the script you can see `--face-encoder` and `--face-encoder-weights-path` parameters. For now they are useless, because we do not compute the third component of the loss function, which would require the differentialbe warping to be implemented. You can ignore these parameters