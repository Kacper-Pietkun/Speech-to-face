import torch.nn as nn


class FaceDecoder(nn.Module):
    """
    Input to the model should be face embedding vector of dimension equal to 4096
    """
    def __init__(self):
        super(FaceDecoder, self).__init__()
        self.pre_layers = self.define_pre_layers()
        self.landmark_layers = self.define_landmark_layers()
        self.texture_layers = self.define_texture_layers()
    

    def define_pre_layers(self):
        layers = nn.ModuleList()
        pre_layer1 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=3072),
            nn.BatchNorm1d(num_features=3072),
            nn.ReLU()
        )
        layers.append(pre_layer1)

        pre_layer2 = nn.Sequential(
            nn.Linear(in_features=3072, out_features=2048),
            nn.BatchNorm1d(num_features=2048),
            nn.ReLU()
        )
        layers.append(pre_layer2)

        return layers


    def define_landmark_layers(self):
        layers = nn.ModuleList()
        landmark_layer1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU()
        )
        layers.append(landmark_layer1)

        landmark_layer2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU()
        )
        layers.append(landmark_layer2)

        landmark_layer3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU()
        )
        layers.append(landmark_layer3)

        landmark_layer4 = nn.Sequential(
            nn.Linear(in_features=256, out_features=144),
        )
        layers.append(landmark_layer4)

        return layers


    def define_texture_layers(self):
        layers = nn.ModuleList()
        texture_layer1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256 * 14 * 14),
            nn.ReLU()
        )
        layers.append(texture_layer1)

        texture_layer2 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(256, 14, 14)),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
        layers.append(texture_layer2)

        texture_layer3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
        layers.append(texture_layer3)

        texture_layer4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
        layers.append(texture_layer4)

        texture_layer5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
        layers.append(texture_layer5)

        texture_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1)
        )
        layers.append(texture_layer6)

        return layers


    def forward(self, x):
        for pre_layer in self.pre_layers:
            x = pre_layer(x)

        landmarks = x
        for landmark_layer in self.landmark_layers:
            landmarks = landmark_layer(landmarks)

        texture = x
        for texture_layer in self.texture_layers:
            texture = texture_layer(texture)

        return landmarks, texture
    
    def get_predifined_layer_activation(self, x):
        for pre_layer in self.pre_layers:
            x = pre_layer(x)
        return x
