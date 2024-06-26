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
from model_saver import ModelSaver
from losses import FaceDecoderLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


parser = ArgumentParser(description="Face Decoder Training")

parser.add_argument("--train-dataset-path", type=str, required=True,
                    help="Absolute path to the dataset train split with face embeddings, face landmarks and face images")

parser.add_argument("--val-dataset-path", type=str, required=True,
                    help="Absolute path to the dataset validation split with face embeddings, face landmarks and face images")

parser.add_argument("--batch-size", type=int, default=16,
                    help="must be bigger than 1, because of the BatchNorm operator")

parser.add_argument("--learning-rate", type=float, default=1e-3)

parser.add_argument("--num-epochs", type=int, default=1000)

parser.add_argument("--gpu", type=int, default=0,
                    help="-1 for cpu training")

parser.add_argument("--face-encoder", type=str, default="vgg_face_serengil", choices=["vgg_face_serengil", "vgg_face_16_rcmalli"],
                    help="Backend for calculating embedding (features) of images")

parser.add_argument("--face-encoder-weights-path", type=str,
                    help="Absolute path to a file where model weights for face encoder backend are stored")

parser.add_argument("--save-folder-path", type=str, required=True,
                    help="Folder were all the result files will be saved")

parser.add_argument("--save-images", action="store_true",
                    help="If set, then save first images at the beginning of each epoch - original and predicted \
                          (one image for training set and one image for validation set)")

parser.add_argument("--continue-training-path", type=str,
                    help="path to the file of the model that will be used to continue training. If not passed then new model will be trained")


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
    if choice is None:
        return None
    elif choice == "vgg_face_serengil":
        model = VGGFace_serengil()
    elif choice == "vgg_face_16_rcmalli":
        model = VGGFace16_rcmalli()
    else:
        raise ValueError(f"Unrecognized backend for face encoder: {choice}")

    model.load_state_dict(torch.load(weights_path))
    return model


def save_face_visualizations(args, images_true, images_predicted, landmarks_true, landmarks_predicted, epoch, dataset_type):
    if args.save_images:
        _, axes = plt.subplots(1, 2, figsize=(12, 8))
        temp_image = (images_true[0].cpu() * 255).to(torch.uint8)
        axes[0].imshow(temp_image.permute(1, 2, 0))
        axes[0].set_title("Original image")
        axes[0].axis("off")
        temp_image = landmarks_true[0].cpu().view(72, 2)
        x, y =zip(*temp_image.squeeze(0))
        axes[0].scatter(x, y, c='red', marker='o', s=2)

        temp_landmark = (images_predicted[0].cpu() * 255).to(torch.uint8)
        axes[1].imshow(temp_landmark.permute(1, 2, 0))
        axes[1].set_title("Predicted image")
        axes[1].axis("off")
        temp_landmark = landmarks_predicted[0].cpu().detach().view(72, 2)
        x, y =zip(*temp_landmark.squeeze(0))
        axes[1].scatter(x, y, c='red', marker='o', s=2)
        plt.tight_layout()
        directory = f"{args.save_folder_path}/images/{dataset_type}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{epoch}.jpg")
        plt.clf()
        plt.close()


def save_history_plots(args, history):
    losses_keys = ["train_sum_loss", "train_landmarks_loss", "train_textures_loss", "train_embeddings_loss",
                   "val_sum_loss", "val_landmarks_loss", "val_textures_loss", "val_embeddings_loss"]
    for loss_key in losses_keys:
        plt.plot(history[loss_key], color='b', label=loss_key)
        plt.title(f'Loss: {loss_key} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        directory = f"{args.save_folder_path}/images"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{loss_key}.jpg")
        plt.clf()
        plt.close()


def run(args, face_decoder, face_encoder, optimizer, loss_fn, model_saver, train_dataloader, val_dataloader, device, start_epoch=0, history=[]):
    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        face_decoder.train()
        train_sum_loss, train_landmarks_loss, train_textures_loss, train_embeddings_loss = 0, 0, 0, 0
        saved_image_index = np.random.randint(0, len(train_dataloader))
        for step, (images_true, embeddings_true, landmarks_true) in enumerate(tqdm(train_dataloader)):
            images_true, embeddings_true, landmarks_true = images_true.to(device), embeddings_true.to(device), landmarks_true.to(device)
            if embeddings_true.shape[0] == 1:
                continue

            optimizer.zero_grad()
            landmarks_predicted, images_predicted = face_decoder(embeddings_true)
            with torch.no_grad():
                embeddings_predicted = face_encoder(images_predicted, get_embedding=True)
            sum_loss, landmarks_loss, textures_loss, embeddings_loss = loss_fn(landmarks_true, landmarks_predicted, images_true, images_predicted,
                                                                               embeddings_true, embeddings_predicted)
            sum_loss.backward()
            optimizer.step()

            train_sum_loss += sum_loss.item() * images_true.size(0)
            train_landmarks_loss += landmarks_loss.item() * images_true.size(0)
            train_textures_loss += textures_loss.item() * images_true.size(0)
            train_embeddings_loss += embeddings_loss.item() * images_true.size(0)
            if step == saved_image_index:
                save_face_visualizations(args, images_true, images_predicted, landmarks_true, landmarks_predicted, epoch, "train")
    
        face_decoder.eval()
        val_sum_loss, val_landmarks_loss, val_textures_loss, val_embeddings_loss = 0, 0, 0, 0
        saved_image_index = np.random.randint(0, len(val_dataloader))
        with torch.no_grad():
            for step, (images_true, embeddings_true, landmarks_true) in enumerate(tqdm(val_dataloader)):
                images_true, embeddings_true, landmarks_true = images_true.to(device), embeddings_true.to(device), landmarks_true.to(device)
                landmarks_predicted, images_predicted = face_decoder(embeddings_true)
                embeddings_predicted = face_encoder(images_predicted, get_embedding=True)
                sum_loss, landmarks_loss, textures_loss, embeddings_loss = loss_fn(landmarks_true, landmarks_predicted, images_true, images_predicted,
                                                                                   embeddings_true, embeddings_predicted)

                val_sum_loss += sum_loss.item() * images_true.size(0)
                val_landmarks_loss += landmarks_loss.item() * images_true.size(0)
                val_textures_loss += textures_loss.item() * images_true.size(0)
                val_embeddings_loss += embeddings_loss.item() * images_true.size(0)

                if step == saved_image_index:
                    save_face_visualizations(args, images_true, images_predicted, landmarks_true, landmarks_predicted, epoch, "val")

        train_sum_loss /= len(train_dataloader.sampler)
        val_sum_loss /= len(val_dataloader.sampler)
        history.append({
            "epoch": epoch,
            "train_sum_loss": train_sum_loss,
            "train_landmarks_loss": train_landmarks_loss / len(train_dataloader.sampler),
            "train_textures_loss": train_textures_loss / len(train_dataloader.sampler),
            "train_embeddings_loss": train_embeddings_loss / len(train_dataloader.sampler),
            "val_sum_loss": val_sum_loss,
            "val_landmarks_loss": val_landmarks_loss / len(val_dataloader.sampler),
            "val_textures_loss": val_textures_loss / len(val_dataloader.sampler),
            "val_embeddings_loss": val_embeddings_loss / len(val_dataloader.sampler)
        })
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{args.save_folder_path}/history.csv", index=False)
        model_saver.save(val_sum_loss, epoch, face_decoder.state_dict(), 
                                   optimizer.state_dict(), history, epoch%10==0)
        save_history_plots(args, history_df)
        print('Epoch: {} Train Loss: {:.4f} Validation Loss: {:.4f} '.format(epoch, train_sum_loss, val_sum_loss))


def main():
    args = parser.parse_args()

    device = get_device(args.gpu)
    train_dataset = FaceDecoderDataset(args.train_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = FaceDecoderDataset(args.val_dataset_path)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    face_decoder = FaceDecoder().to(device)
    face_encoder = None
    if args.face_encoder_weights_path:
        face_encoder = get_face_encoder(args.face_encoder, args.face_encoder_weights_path).to(device)
        face_encoder.eval()

    optimizer = optim.Adam(face_decoder.parameters(), lr=args.learning_rate)
    loss_fn = FaceDecoderLoss()
    
    if args.continue_training_path is None:
        # Train new model
        model_saver = ModelSaver(f"{args.save_folder_path}/latest_model.pt",
                            f"{args.save_folder_path}/best_model.pt")
        run(args, face_decoder, face_encoder, optimizer, loss_fn, model_saver, train_dataloader, val_dataloader, device)
    else:
        # Continue training existing model
        checkpoint = torch.load(args.continue_training_path)
        epoch = checkpoint["epoch"] + 1
        model_saver = ModelSaver(f"{args.save_folder_path}/latest_model.pt",
                            f"{args.save_folder_path}/best_model.pt",
                            checkpoint["best_loss"])
        face_decoder.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        run(args, face_decoder, face_encoder, optimizer, loss_fn, model_saver, train_dataloader, val_dataloader, device, epoch, history)


if __name__ == "__main__":
    main()
