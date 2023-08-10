import torch.nn as nn

class VGGFace_serengil(nn.Module):
    def __init__(self):
        super(VGGFace_serengil, self).__init__()

        self.layers = []

        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer1)

        self.layer2 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer2)

        self.layer3 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer3)

        self.layer4 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer4)

        self.layer5 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer5)

        self.layer6 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer6)

        self.layer7 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer7)

        self.layer8 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer8)

        self.layer9 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer9)

        self.layer10 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer10)

        self.layer11 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer11)

        self.layer12 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1),
            nn.ReLU()
        )
        self.layers.append(self.layer12)

        self.layer13 = nn.Sequential(
            nn.ZeroPad2d(padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer13)

        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=(7,7), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.layers.append(self.layer14)

        self.layer15 = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.layers.append(self.layer15)

        self.layer16 = nn.Sequential(
            nn.Conv2d(in_channels=4096, out_channels=2622, kernel_size=(1,1), stride=1),
            nn.Flatten(),
            nn.Softmax()
        )
        self.layers.append(self.layer16)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VGGFace16_rcmalli(nn.Module):
    def __init__(self):
        super(VGGFace16_rcmalli, self).__init__()

        self.layers = []

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer3)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer4)

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer5)

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer6)

        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer7)

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer8)

        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer9)

        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer10)

        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer11)

        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU()
        )
        self.layers.append(self.layer12)

        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.layers.append(self.layer13)

        self.layer14 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU()
        )
        self.layers.append(self.layer14)

        self.layer15 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
        )
        self.layers.append(self.layer15)

        self.layer16 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2622),
            nn.Softmax()
        )
        self.layers.append(self.layer16)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
