# Face Encoder
In our project we allow user to choose any of the following implementations of the face encoders:
- VGG-face from https://github.com/serengil/deepface

These models are written and trained in Tensorflow, however, in our project we use PyTorch. So, there is a need to convert models from one framework to another.
In `tensorflow_weights` folder you can find Tensorflow weights for each model listed above (Weights were downloaded from repositories listed above). You can run following scripts:
- `VGGFace_serengil.py` to convert weights of VGG-face model from https://github.com/serengil/deepface from Tensorflow to Pytorch

# Note
All of the weights have already been converted to PyTorch format and saved in a repository under directory `src/models/pytorch_weights`. You don't need to do it manually. This folder was made public for people who are curious how we converted models between Tensorflow and PyTorch. 