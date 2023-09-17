import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.linalg import norm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from datasets.s2f_dataset import S2fDataset
from models.face_encoder import VGGFace16_rcmalli, VGGFace_serengil
from models.face_decoder import FaceDecoder
from models.voice_encoder import VoiceEncoder
from tqdm import tqdm
import pandas as pd


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

parser.add_argument("--save-folder-path", type=str, required=True,
                    help="Folder were all the result files will be saved")

parser.add_argument("--continue-training-path", type=str,
                    help="path to the file of the model that will be used to continue training. If not passed then new model will be trained")


class S2FLoss(nn.Module):
    def __init__(self, face_encoder_last_layer, face_decoder_first_layer, coe_1=0.025, coe_2=200):
        super().__init__()
        self.face_encoder_last_layer = face_encoder_last_layer
        self.face_decoder_first_layer = face_decoder_first_layer
        self.mae_loss = nn.L1Loss()
        self.coe_1 = coe_1
        self.coe_2 = coe_2

    def forward(self, pred, true):
        sum_loss = 0
        loss_base, loss_face_encoder, loss_face_decoder = 0, 0, 0

        # loss_base - v_f and v_s distance part
        vs_normalized = pred / norm(pred)
        vf_normalized = true / norm(true)
        loss_base = torch.pow(norm(vf_normalized - vs_normalized), 2)
        loss_base *= self.coe_1
        sum_loss += loss_base

        # loss_face_encoder - face encoder last layer activation part
        vgg_v_s = self.face_encoder_last_layer(pred)
        with torch.no_grad():
            vgg_v_f = self.face_encoder_last_layer(true)
        loss_face_encoder = self.knowledge_distilation(vgg_v_f, vgg_v_s)
        loss_face_encoder *= self.coe_2
        sum_loss += loss_face_encoder

        # loss_face_decoder - face decoder first layers activation part
        dec_v_s = self.face_decoder_first_layer(pred)
        with torch.no_grad():
            dec_v_f = self.face_decoder_first_layer(true)
            loss_face_decoder = self.mae_loss(dec_v_f, dec_v_s)
            sum_loss += loss_face_decoder

        return sum_loss, loss_base, loss_face_encoder, loss_face_decoder
    
    def knowledge_distilation(self, a, b, T=2):
        p_a = F.softmax(a / T, dim=1)
        p_b = F.log_softmax(b / T, dim=1)
        return -(p_a * p_b).sum()


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


def run(args, voice_encoder, optimizer, loss_fn, dataloader, device):
    history = []
    for epoch in range(args.num_epochs):
        train_sum_loss, train_base_loss, train_face_encoder_loss, train_face_decoder_loss = 0, 0, 0, 0
        voice_encoder.train()
        for (inputs, true_embeddings) in tqdm(dataloader):
            inputs = inputs.to(device)
            true_embeddings = true_embeddings.to(device)

            optimizer.zero_grad()

            voice_encoder_embeddings = voice_encoder(inputs)
            sum_loss, base_loss, face_encoder_loss, face_decoder_loss  = loss_fn(voice_encoder_embeddings, true_embeddings)
            sum_loss.backward()
            optimizer.step()

            train_sum_loss += sum_loss.item() * inputs.size(0)
            train_base_loss += base_loss.item() * inputs.size(0)
            train_face_encoder_loss += face_encoder_loss.item() * inputs.size(0)
            train_face_decoder_loss += face_decoder_loss.item() * inputs.size(0)
    
        train_sum_loss /= len(dataloader.sampler)
        history.append({"epoch": epoch,
                        "train_sum_loss": train_sum_loss,
                        "train_base_loss": train_base_loss / len(dataloader.sampler),
                        "train_face_encoder_loss": train_face_encoder_loss / len(dataloader.sampler),
                        "train_face_decoder_loss": train_face_decoder_loss / len(dataloader.sampler)})
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{args.save_folder_path}/history.csv", index=False)
        print('Epoch: {} Train Loss: {:.4f} '.format(epoch, train_sum_loss))


def main():
    args = parser.parse_args()

    device = get_device(args.gpu)

    train_dataset = S2fDataset(args.dataset_path)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    voice_encoder = VoiceEncoder().to(device)
    face_decoder = FaceDecoder().to(device)
    face_decoder.eval()
    face_encoder = get_face_encoder(args.face_encoder, args.face_encoder_weights_path).to(device)
    face_encoder.eval()

    optimizer = optim.Adam(voice_encoder.parameters(), lr=args.learning_rate)
    loss_fn = S2FLoss(face_encoder.get_last_layer_activation, face_decoder.get_predifined_layer_activation)

    run(args, voice_encoder, optimizer, loss_fn, dataloader, device)


if __name__ == "__main__":
    main()
