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
from datasets.s2f_dataset_all_to_all import S2fDatasetAlltoAll
from datasets.s2f_dataset_one_to_one import S2fDatasetOneToOne
from datasets.s2f_dataset_all_to_one import S2fDatasetAllToOne
from models.face_encoder import VGGFace16_rcmalli, VGGFace_serengil
from models.face_decoder import FaceDecoder
from model_saver import ModelSaver
from losses import S2FLoss
from early_stopper import EarlyStopper
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoModelForAudioClassification


parser = ArgumentParser(description="Voice Encoder Training")

parser.add_argument("--train-dataset-path", type=str, required=True,
                    help="Absolute path to the dataset train split with face embeddings and spectrograms")

parser.add_argument("--val-dataset-path", type=str, required=True,
                    help="Absolute path to the dataset validation split with face embeddings and spectrograms")

parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--batch-size-fine-tune", type=int, default=2)

parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument("--learning-rate-fine-tune", type=float, default=0.000001)

parser.add_argument("--num-epochs", type=int, default=100)

parser.add_argument("--gpu", type=int, default=0,
                    help="-1 for cpu training")

parser.add_argument("--num-workers", type=int, default=12,
                    help="Number of workers for the train and validation dataloaders")

parser.add_argument("--dataloader-type", type=str, default="all_to_one", choices=["all_to_all", "one_to_one", "all_to_one"],
                    help="Type of the dataloader (all_to_all takes much more time, because it creates much more training pairs)")

parser.add_argument("--face-encoder", type=str, default="vgg_face_serengil", choices=["vgg_face_serengil", "vgg_face_16_rcmalli"],
                    help="Backend for calculating embedding (features) of images")

parser.add_argument("--face-encoder-weights-path", required=True, type=str,
                    help="Absolute path to a file where model weights for face encoder model are stored")

parser.add_argument("--face-decoder-weights-path", required=True, type=str,
                    help="Absolute path to a file where model weights for face decoder model are stored")

parser.add_argument("--save-folder-path", type=str, required=True,
                    help="Folder were all the result files will be saved")

parser.add_argument("--save-images", action="store_true",
                    help="If set, then save first images at the beginning of each epoch - original and predicted \
                            (one image for training set and one image for validation set). Saved images are the images \
                            reconstructed from the output of the voice encoder by the face decoder")

parser.add_argument("--continue-training-path", type=str,
                    help="path to the file of the model that will be used to continue training. If not passed then new model will be trained")

parser.add_argument("--early-stopping", action="store_true",
                    help="Use early stopping")

parser.add_argument("--early-stopping-patience", type=int, default=5,
                    help="Patience for early stopping - allowed number of epochs with no improvement")

parser.add_argument("--fine-tune", action="store_true",
                    help="unfreeze and train model")

parser.add_argument("--unfreeze-number", type=int, default=-1,
                    help="determines from which layer model should be unfrozen (-1 means unfreeze all layers)")


def get_device(choice):
    if choice >= 0:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{choice}")
            torch.backends.cudnn.benchmark = True
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


def save_face_visualizations(args, face_decoder, voice_encoder_embeddings, epoch, dataset_type):
    if args.save_images:

        with torch.no_grad():
            landmarks_predicted, images_predicted = face_decoder(voice_encoder_embeddings)

        _, axes = plt.subplots(1, 1, figsize=(12, 8))
        temp_image = (images_predicted[0].cpu() * 255).to(torch.uint8)
        axes.imshow(temp_image.permute(1, 2, 0))
        axes.set_title("Reconstructed image")
        axes.axis("off")
        # temp_image = landmarks_predicted[0].cpu().view(72, 2)
        # x, y =zip(*temp_image.squeeze(0))
        # axes.scatter(x, y, c='red', marker='o', s=2)

        plt.tight_layout()
        directory = f"{args.save_folder_path}/images/{dataset_type}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/{epoch}.jpg")
        plt.clf()
        plt.close()


def save_history_plots(args, history):
    losses_keys = ["train_sum_loss", "train_base_loss", "train_face_encoder_loss", "train_face_decoder_loss",
                   "val_sum_loss", "val_base_loss", "val_face_encoder_loss", "val_face_decoder_loss"]
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


def run(args, ast, optimizer, loss_fn, model_saver, early_stopper, train_dataloader, val_dataloader, face_decoder, device, start_epoch=0, history=[]):
    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        ast.train()
        train_sum_loss, train_base_loss, train_face_encoder_loss, train_face_decoder_loss = 0, 0, 0, 0
        saved_image_index = np.random.randint(0, len(train_dataloader))
        for step, (inputs, true_embeddings) in enumerate(tqdm(train_dataloader)):
            inputs, true_embeddings = inputs.to(device), true_embeddings.to(device)

            optimizer.zero_grad()
            voice_encoder_embeddings = ast(inputs)
            voice_encoder_embeddings = voice_encoder_embeddings.logits
            sum_loss, base_loss, face_encoder_loss, face_decoder_loss  = loss_fn(voice_encoder_embeddings, true_embeddings)
            sum_loss.backward()
            optimizer.step()

            train_sum_loss += sum_loss.item() * inputs.size(0)
            train_base_loss += base_loss.item() * inputs.size(0)
            train_face_encoder_loss += face_encoder_loss.item() * inputs.size(0)
            train_face_decoder_loss += face_decoder_loss.item() * inputs.size(0)
            if step == saved_image_index:
                save_face_visualizations(args, face_decoder, voice_encoder_embeddings, epoch, "train")
    
        ast.eval()
        val_sum_loss, val_base_loss, val_face_encoder_loss, val_face_decoder_loss = 0, 0, 0, 0
        saved_image_index = np.random.randint(0, len(val_dataloader))
        with torch.no_grad():
            for step, (inputs, true_embeddings) in enumerate(tqdm(val_dataloader)):
                inputs, true_embeddings = inputs.to(device), true_embeddings.to(device)
                voice_encoder_embeddings = ast(inputs)
                voice_encoder_embeddings = voice_encoder_embeddings.logits
                sum_loss, base_loss, face_encoder_loss, face_decoder_loss  = loss_fn(voice_encoder_embeddings, true_embeddings)

                val_sum_loss += sum_loss.item() * inputs.size(0)
                val_base_loss += base_loss.item() * inputs.size(0)
                val_face_encoder_loss += face_encoder_loss.item() * inputs.size(0)
                val_face_decoder_loss += face_decoder_loss.item() * inputs.size(0)

                if step == saved_image_index:
                    save_face_visualizations(args, face_decoder, voice_encoder_embeddings, epoch, "val")

        train_sum_loss /= len(train_dataloader.sampler)
        val_sum_loss /= len(val_dataloader.sampler)
        history.append({
            "epoch": epoch,
            "train_sum_loss": train_sum_loss,
            "train_base_loss": train_base_loss / len(train_dataloader.sampler),
            "train_face_encoder_loss": train_face_encoder_loss / len(train_dataloader.sampler),
            "train_face_decoder_loss": train_face_decoder_loss / len(train_dataloader.sampler),
            "val_sum_loss": val_sum_loss,
            "val_base_loss": val_base_loss / len(val_dataloader.sampler),
            "val_face_encoder_loss": val_face_encoder_loss / len(val_dataloader.sampler),
            "val_face_decoder_loss": val_face_decoder_loss / len(val_dataloader.sampler)
        })
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{args.save_folder_path}/history.csv", index=False)
        model_saver.save(val_sum_loss, epoch, ast.state_dict(), 
                                   optimizer.state_dict(), history, epoch%1==0)
        save_history_plots(args, history_df)
        print('Epoch: {} Train Loss: {:.4f} Validation Loss: {:.4f} '.format(epoch, train_sum_loss, val_sum_loss))

        if early_stopper:
            decision = early_stopper.should_stop(val_sum_loss)
            if decision is True:
                print("EARLY STOPPER - STOP TRAINING")
                return


def main():
    args = parser.parse_args()

    device = get_device(args.gpu)

    print(f"Dataloader: {args.dataloader_type}")
    if args.dataloader_type == "all_to_all":
        train_dataset = S2fDatasetAlltoAll(args.train_dataset_path, is_ast=True)
        val_dataset = S2fDatasetAlltoAll(args.val_dataset_path, is_ast=True)
    elif args.dataloader_type == "one_to_one":
        train_dataset = S2fDatasetOneToOne(args.train_dataset_path, is_ast=True)
        val_dataset = S2fDatasetOneToOne(args.val_dataset_path, is_ast=True)
    elif args.dataloader_type == "all_to_one":
        train_dataset = S2fDatasetAllToOne(args.train_dataset_path, is_ast=True)
        val_dataset = S2fDatasetAllToOne(args.val_dataset_path, is_ast=True)
    else:
        raise ValueError(f"Dataloader '{args.dataloader_type}' does not exist")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ast = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=4096, ignore_mismatched_sizes=True).to(device)
    head = ast.classifier
    new_head = nn.Sequential(
        head,
        nn.ReLU()
    )
    ast.classifier = new_head
    # freeze every layer but - classifier.dense.bias and classifier.dense.weight
    for name, param in ast.named_parameters():
        if name != "classifier.0.dense.weight" and name != "classifier.0.dense.bias":
            param.requires_grad = False
        else:
            nn.init.trunc_normal_(param)

    face_decoder = FaceDecoder().to(device)
    face_decoder_checkpoint = torch.load(args.face_decoder_weights_path)
    face_decoder.load_state_dict(face_decoder_checkpoint["model_state_dict"])
    face_decoder.eval()

    face_encoder = get_face_encoder(args.face_encoder, args.face_encoder_weights_path).to(device)
    face_encoder.eval()

    print(f"lr: {args.learning_rate}")
    optimizer = optim.Adam(ast.parameters(), lr=args.learning_rate)
    loss_fn = S2FLoss(face_encoder.get_last_layer_activation, face_decoder.get_predifined_layer_activation, coe_1=30, coe_2=2, coe_3=50)

    early_stopper = None
    if args.early_stopping:
        early_stopper = EarlyStopper(args.early_stopping_patience)

    if args.continue_training_path is None:
        print("HEAD TRAINING")
        # Train new model
        model_saver = ModelSaver(f"{args.save_folder_path}/latest_model.pt",
                            f"{args.save_folder_path}/best_model.pt")
        history = run(args, ast, optimizer, loss_fn, model_saver, early_stopper, train_dataloader, val_dataloader, face_decoder, device)
    elif args.continue_training_path is not None and args.fine_tune is None:
        print("CONTINUE HEAD TRAINING")
        # Continue training existing model
        checkpoint = torch.load(args.continue_training_path)
        epoch = checkpoint["epoch"] + 1
        model_saver = ModelSaver(f"{args.save_folder_path}/latest_model.pt",
                                 f"{args.save_folder_path}/best_model.pt",
                                 checkpoint["best_loss"])
        ast.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        history = run(args, ast, optimizer, loss_fn, model_saver, early_stopper, train_dataloader, val_dataloader, face_decoder, device, epoch, history)
    elif args.continue_training_path is not None and args.fine_tune is not None:
        print("FINE TUNNING")
        checkpoint = torch.load(args.continue_training_path)
        epoch = checkpoint["epoch"] + 1
        model_saver = ModelSaver(f"{args.save_folder_path}/latest_model.pt",
                                 f"{args.save_folder_path}/best_model.pt",
                                 checkpoint["best_loss"])
        ast.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        
        for name, param in ast.named_parameters():
            param.requires_grad = True
        if args.unfreeze_number != -1:
            counter = 1
            for name, param in ast.named_parameters():
                if counter <= args.unfreeze_number:
                    param.requires_grad = False
                counter += 1
        for name, param in ast.named_parameters():
            print(f"{name}, {param.requires_grad}")
        
        print(f"learning_rate_fine_tune: {args.learning_rate_fine_tune}")
        optimizer = optim.Adam(ast.parameters(), lr=args.learning_rate_fine_tune)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_fine_tune, shuffle=True, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size_fine_tune, shuffle=False, num_workers=args.num_workers)
        
        model_saver.start_finetune(epoch)

        history = run(args, ast, optimizer, loss_fn, model_saver, early_stopper, train_dataloader, val_dataloader, face_decoder, device, epoch, history)
    else:
        print("ERROR - wrong configuration")



if __name__ == "__main__":
    main()
