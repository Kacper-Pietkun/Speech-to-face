import torch.nn as nn


class VoiceEncoder(nn.Module):
    """
    Input to the model should be an spectrogram calculated from an audio file 6 seconds - with shape (2, 257, 601)
    """
    def __init__(self):
        super(VoiceEncoder, self).__init__()

        self.layers = []
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layers.append(self.layer1)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.layers.append(self.layer2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.layers.append(self.layer3)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.layers.append(self.layer4)

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.layers.append(self.layer5)

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        self.layers.append(self.layer6)

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layers.append(self.layer7)

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layers.append(self.layer8)

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2),
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        self.layers.append(self.layer9)

        self.layer10 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=29184, out_features=4096),
            nn.ReLU()
        )
        self.layers.append(self.layer10)

        self.layer11 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        )
        self.layers.append(self.layer11)
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
