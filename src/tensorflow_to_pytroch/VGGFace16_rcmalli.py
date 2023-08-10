import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from argparse import ArgumentParser
from models.face_encoder import VGGFace16_rcmalli
import numpy as np
import torch
import torch.nn as nn
from keras import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense
from keras.backend import image_data_format, set_image_data_format

parser = ArgumentParser(description="Getting features out of images of faces")

parser.add_argument("--weights-path", required=True, type=str,
                    help="Absolute path to the file, where tensorflow model weights are located")

parser.add_argument("--save-path", required=True, type=str,
                    help="Absolute path to the file, where pytorch model's weights will be saved")


def get_tensorflow_VGGFace16_rcmalli(args):

    img_input = Input(shape=(224, 224, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, name='fc6')(x)
    x = Activation('relu', name='fc6/relu')(x)
    x = Dense(4096, name='fc7')(x)
    x = Activation('relu', name='fc7/relu')(x)
    x = Dense(2622, name='fc8')(x)
    x = Activation('softmax', name='fc8/softmax')(x)

    # Create model.
    model = Model(img_input, x, name='vggface_vgg16')
    model.load_weights(args.weights_path, by_name=True)

    return model


def transfer_weights_from_tensorflow_to_pytorch(model_tf, model_pt):
    layers_pt = model_pt.modules()
    layers_tf = model_tf.layers
    layers_tf = (layer for layer in layers_tf)

    while True:
        layer_pt = next(layers_pt, None)
        while layer_pt is not None and not isinstance(layer_pt, nn.Conv2d) and not isinstance(layer_pt, nn.Linear):
            layer_pt = next(layers_pt, None)

        layer_tf = next(layers_tf, None)
        while layer_tf is not None and not isinstance(layer_tf, Conv2D) and not isinstance(layer_tf, Dense):
            layer_tf = next(layers_tf, None)

        if layer_pt is None or layer_tf is None:
            break

        weight_tf = layer_tf.weights[0]
        bias_tf = layer_tf.bias.numpy()

        if isinstance(layer_pt, nn.Conv2d):
            weight_tf = np.transpose(weight_tf, (3, 2, 0, 1))   
        elif isinstance(layer_pt, nn.Linear):
            weight_tf = np.transpose(weight_tf, (1, 0))

        layer_pt.weight.data = torch.from_numpy(weight_tf)
        layer_pt.bias.data = torch.from_numpy(bias_tf) 


def main():
    args = parser.parse_args()

    if image_data_format() == "channels_first":
        set_image_data_format("channels_last")

    model_tf = get_tensorflow_VGGFace16_rcmalli(args)
    model_pt = VGGFace16_rcmalli()

    transfer_weights_from_tensorflow_to_pytorch(model_tf, model_pt)
    
    torch.save(model_pt.state_dict(), args.save_path)


if __name__ == "__main__":
    main()
