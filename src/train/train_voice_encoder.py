import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.linalg import norm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from datasets.s2f_dataset import S2fDataset
from models.face_encoder import VGGFace16_rcmalli, VGGFace_serengil
from models.voice_encoder import VoiceEncoder
import time
from tqdm import tqdm


parser = ArgumentParser(description="Voice Encoder Training")

parser.add_argument("--dataset-path", type=str, required=True,
                    help="Absolute path to the dataset with face embeddings and spectrograms")

parser.add_argument("--batch-size", type=int, default=2)

parser.add_argument("--learning-rate", type=float, default=1e-3)

parser.add_argument("--num-epochs", type=int, default=100)

parser.add_argument("--gpu", type=int, default=0,
                    help="-1 for cpu training")

parser.add_argument("--face-encoder", type=str, default="vgg_face_serengil", choices=["vgg_face_serengil", "vgg_face_16_rcmalli"],
                    help="Backend for calculating embedding (features) of images")

parser.add_argument("--face-encoder-weights-path", required=True, type=str,
                    help="Absolute path to a file where model weights for face encoder backend are stored")

parser.add_argument("--save-model-path", type=str, default = "./best_voce_encoder.pt",
                    help="Path to the file where best model's state dict will be saved")


class S2FLoss(nn.Module):
    def __init__(self, coe_1=0.025, coe_2=200):
        super().__init__()
        self.coe_1 = coe_1
        self.coe_2 = coe_2

    def forward(self, pred, true):
        # TODO: To loss add:
        # - the difference in the activation of the last layer of the face encoder, f_VGG : R^4096 -> R^2622
        # - the difference in the activation of the first layer of the face decoder, f_dec : R^4096 -> R^1000

        vs_normalized = pred / norm(pred)
        vf_normalized = true / norm(true)
        loss = torch.pow(norm(vf_normalized - vs_normalized), 2)
        loss *= self.coe_1
        return loss


def get_device(choice):
    if choice >= 0:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{choice}")
        else:
            raise ValueError("GPU training was chosen but cuda is not available")
    else:
        device = torch.device("cpu")
    return device
    

def get_face_encoder(choice, weights_path):
    if choice == "vgg_face_serengil":
        model = VGGFace_serengil()
    elif choice == "vgg_face_16_rcmalli":
        model = VGGFace16_rcmalli()
    else:
        raise ValueError(f"Unrecognized backend for face encoder: {choice}")

    model.load_state_dict(torch.load(weights_path))
    return model


def run(args, voice_encoder, face_encoder, optimizer, loss_fn, dataloader, device):

    face_encoder.eval()
    for epoch in range(args.num_epochs):
        train_loss = 0
        voice_encoder.train()
        for (inputs, outputs) in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            optimizer.zero_grad()

            voice_encoder_embedding = voice_encoder(inputs)
            loss = loss_fn(voice_encoder_embedding, outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
    
        train_loss /= len(dataloader.sampler)
        print('Epoch: {} Train Loss: {:.4f} '.format(epoch, train_loss))


def main():
    args = parser.parse_args()

    device = get_device(args.gpu)

    train_dataset = S2fDataset(args.dataset_path)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    voice_encoder = VoiceEncoder().to(device)
    face_encoder = get_face_encoder(args.face_encoder, args.face_encoder_weights_path).to(device)

    optimizer = optim.Adam(voice_encoder.parameters(), lr=args.learning_rate)
    loss_fn = S2FLoss()

    run(args, voice_encoder, face_encoder, optimizer, loss_fn, dataloader, device)


if __name__ == "__main__":
    main()
