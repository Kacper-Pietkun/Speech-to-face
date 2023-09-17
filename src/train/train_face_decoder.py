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
import matplotlib.pyplot as plt
import pandas as pd


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

parser.add_argument("--save-folder-path", type=str, required=True,
                    help="Folder were all the result files will be saved")

parser.add_argument("--save-images", type=bool, default=False,
                    help="If true, then save two first images at the beginning of each epoch - original and predicted")


class FaceDecoderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cos_loss = nn.CosineEmbeddingLoss()

    def forward(self, landmarks_true, landmarks_predicted,
                textures_true, textures_predicted,
                embeddings_true=None, embeddings_predicted=None):
        sum_loss = 0
        loss_landmarks, loss_textures, loss_embeddings = torch.zeros(1), torch.zeros(1), torch.zeros(1)

        # MSE for landmarks
        if landmarks_true is not None and landmarks_predicted is not None:
            loss_landmarks = self.mse_loss(landmarks_true, landmarks_predicted)
            sum_loss += loss_landmarks
        # MAE for textures
        if textures_true is not None and textures_predicted is not None:
            loss_textures = self.mae_loss(textures_true, textures_predicted)
            sum_loss += loss_textures
        # Cosine Similarity loss for embeddings
        if embeddings_true is not None and embeddings_predicted is not None:
            loss_embeddings = self.cos_loss(embeddings_true, embeddings_predicted)
            sum_loss += loss_embeddings

        return sum_loss, loss_landmarks, loss_textures, loss_embeddings


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


def save_images(args, images_true, images_predicted, landmarks_true, landmarks_predicted, epoch, count=2):
    if args.save_images:
        for i in range(count):
            fix, axes = plt.subplots(1, 2, figsize=(12, 8))
            xd = (images_true[i].cpu() * 255).to(torch.uint8)
            axes[0].imshow(xd.permute(1, 2, 0))
            axes[0].set_title("Original image")
            axes[0].axis("off")
            xd = landmarks_true[i].cpu().view(72, 2)
            x, y =zip(*xd.squeeze(0))
            axes[0].scatter(x, y, c='red', marker='o', s=2)

            xd = (images_predicted[i].cpu() * 255).to(torch.uint8)
            axes[1].imshow(xd.permute(1, 2, 0))
            axes[1].set_title("Predicted image")
            axes[1].axis("off")
            xd = landmarks_predicted[i].cpu().detach().view(72, 2)
            x, y =zip(*xd.squeeze(0))
            axes[1].scatter(x, y, c='red', marker='o', s=2)
            plt.tight_layout()
            directory = f"{args.save_folder_path}/images"
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f"{directory}/{epoch}_{i}.jpg")


def run(args, face_decoder, face_encoder, optimizer, loss_fn, dataloader, device):
    history = []
    face_encoder.eval()
    for epoch in range(args.num_epochs):
        train_sum_loss, train_landmarks_loss, train_textures_loss, train_embeddings_loss = 0, 0, 0, 0
        face_decoder.train()
        saved_images = False
        for (images_true, embeddings_true, landmarks_true) in tqdm(dataloader):
            images_true = images_true.to(device)
            embeddings_true = embeddings_true.to(device)
            landmarks_true = landmarks_true.to(device)

            optimizer.zero_grad()
            landmarks_predicted, images_predicted = face_decoder(embeddings_true)
            sum_loss, landmarks_loss, textures_loss, embeddings_loss = loss_fn(landmarks_true, landmarks_predicted,
                                                                                images_true, images_predicted)
            sum_loss.backward()
            optimizer.step()

            train_sum_loss += sum_loss.item() * images_true.size(0)
            train_landmarks_loss += landmarks_loss.item() * images_true.size(0)
            train_textures_loss += textures_loss.item() * images_true.size(0)
            train_embeddings_loss += embeddings_loss.item() * images_true.size(0)
            if saved_images is False:
                saved_images = True
                save_images(args, images_true, images_predicted, landmarks_true, landmarks_predicted, epoch)
    
        history.append({"epoch": epoch,
                        "train_sum_loss": train_sum_loss / len(dataloader.sampler),
                        "train_landmarks_loss": train_landmarks_loss / len(dataloader.sampler),
                        "train_textures_loss": train_textures_loss / len(dataloader.sampler),
                        "train_embeddings_loss": train_embeddings_loss / len(dataloader.sampler)})
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{args.save_folder_path}/history.csv", index=False)
        print('Epoch: {} Train Loss: {:.4f} '.format(epoch, train_sum_loss / len(dataloader.sampler)))


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
