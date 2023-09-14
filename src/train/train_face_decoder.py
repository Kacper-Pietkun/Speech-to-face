import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from datasets.face_decoder_dataset import FaceDecoderDataset
from models.face_encoder import VGGFace16_rcmalli, VGGFace_serengil
from models.face_decoder import FaceDecoder
from tqdm import tqdm

parser = ArgumentParser(description="Face Decoder Training")

parser.add_argument("--dataset-path", type=str, required=True,
                    help="Absolute path to the dataset with face embeddings, face landmarks and face images")

parser.add_argument("--batch-size", type=int, default=2)

parser.add_argument("--learning-rate", type=float, default=1e-3)

parser.add_argument("--num-epochs", type=int, default=100)

parser.add_argument("--gpu", type=int, default=0,
                    help="-1 for cpu training")

parser.add_argument("--face-encoder", type=str, default="vgg_face_serengil", choices=["vgg_face_serengil", "vgg_face_16_rcmalli"],
                    help="Backend for calculating embedding (features) of images")

parser.add_argument("--face-encoder-weights-path", required=True, type=str,
                    help="Absolute path to a file where model weights for face encoder backend are stored")

parser.add_argument("--save-model-path", type=str, default = "./best_face_decoder.pt",
                    help="Path to the file where best model's state dict will be saved")


class FaceDecoderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()

    def forward(self, landmarks_true, landmarks_predicted,
                textures_true, textures_predicted,
                embeddings_true=None, embeddings_predicted=None):
        # MSE for landmarks
        if landmarks_true is not None and landmarks_predicted is not None:
            loss = self.mse_loss(landmarks_true, landmarks_predicted)
        # MAE for textures
        if textures_true is not None and textures_predicted is not None:
            loss += self.mae_loss(textures_true, textures_predicted)
        # Cosine Similarity loss for embeddings
        if embeddings_true is not None and embeddings_predicted is not None:
            loss += self.cos_loss(embeddings_true, embeddings_predicted)
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


def run(args, face_decoder, face_encoder, optimizer, loss_fn, dataloader, device):

    face_encoder.eval()
    for epoch in range(args.num_epochs):
        train_loss = 0
        face_decoder.train()
        for (images_true, embeddings_true, landmarks_true) in tqdm(dataloader):
            images_true = images_true.to(device)
            embeddings_true = embeddings_true.to(device)
            landmarks_true = landmarks_true.to(device)

            optimizer.zero_grad()

            landmarks_predicted, images_predicted = face_decoder(embeddings_true)
            loss = loss_fn(landmarks_true, landmarks_predicted,
                            images_true, images_predicted)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images_true.size(0)
    
        train_loss /= len(dataloader.sampler)
        print('Epoch: {} Train Loss: {:.4f} '.format(epoch, train_loss))


def main():
    args = parser.parse_args()

    device = get_device(args.gpu)

    train_dataset = FaceDecoderDataset(args.dataset_path)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    face_decoder = FaceDecoder().to(device)
    face_encoder = get_face_encoder(args.face_encoder, args.face_encoder_weights_path).to(device)

    optimizer = optim.Adam(face_decoder.parameters(), lr=args.learning_rate)
    loss_fn = FaceDecoderLoss()

    run(args, face_decoder, face_encoder, optimizer, loss_fn, dataloader, device)


if __name__ == "__main__":
    main()
