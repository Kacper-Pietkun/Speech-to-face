# Face Encoder
In our project we allow user to choose any of the following implementations of the face encoders:
- VGG-face from https://github.com/serengil/deepface
- VGGFace16 from https://github.com/rcmalli/keras-vggface

These models are written and trained in Tensorflow, however, in our project we use PyTorch. So, there is a need to convert models from one framework to another.
You can download TensorFlow weights for each model from the repositories listed above. Then, you can run following scripts:
- `VGGFace_serengil.py` to convert weights of VGG-face model from https://github.com/serengil/deepface from Tensorflow to Pytorch
- `VGGFace16_rcmalli.py` to convert weights of VGG-face model from https://github.com/rcmalli/keras-vggface from Tensorflow to Pytorch

# Note
We decided not to place the converted weights here in the repo, because they are too big. If you need any help, we can place the weights on a google drive