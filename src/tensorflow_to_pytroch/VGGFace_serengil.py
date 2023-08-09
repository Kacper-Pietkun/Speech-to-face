import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from argparse import ArgumentParser
from models.face_encoder import VGGFace_serengil
import numpy as np
import torch
import torch.nn as nn
from keras.models import Model, Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.backend import image_data_format, set_image_data_format

parser = ArgumentParser(description="Getting features out of images of faces")

parser.add_argument("--weights-path", required=True, type=str,
                    help="Absolute path to the file, where tensorflow model weights are located")

parser.add_argument("--save-path", required=True, type=str,
                    help="Absolute path to the file, where pytorch model's weights will be saved")


def get_tensorflow_VGGFace_serengil():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model

def transfer_weights_from_tensorflow_to_pytorch(model_tf, model_pt):
    layers_pt = model_pt.modules()
    layers_tf = model_tf.layers
    layers_tf = (layer for layer in layers_tf)

    while True:
        layer_pt = next(layers_pt, None)
        while layer_pt is not None and not isinstance(layer_pt, nn.Conv2d):
            layer_pt = next(layers_pt, None)

        layer_tf = next(layers_tf, None)
        while layer_tf is not None and not isinstance(layer_tf, Convolution2D):
            layer_tf = next(layers_tf, None)

        if layer_pt is None or layer_tf is None:
            break
        weight_tf = layer_tf.weights[0]
        bias_tf = layer_tf.bias.numpy()
        weight_tf = np.transpose(weight_tf, (3, 2, 0, 1))
        
        layer_pt.weight.data = torch.from_numpy(weight_tf)
        layer_pt.bias.data = torch.from_numpy(bias_tf)


def main():
    args = parser.parse_args()

    if image_data_format() == "channels_first":
        set_image_data_format("channels_last")

    model_tf = get_tensorflow_VGGFace_serengil()
    model_tf.load_weights(args.weights_path)
    model_tf = Model(inputs=model_tf.layers[0].input, outputs=model_tf.layers[-2].output)
    model_pt = VGGFace_serengil()

    transfer_weights_from_tensorflow_to_pytorch(model_tf, model_pt)
    
    torch.save(model_pt.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
