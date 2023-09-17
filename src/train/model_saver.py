import torch
import math


class ModelSaver():
    def __init__(self, save_latest_path, save_best_path, best_loss=math.inf):
        self.best_loss = best_loss
        self.save_latest_path = save_latest_path 
        self.save_best_path = save_best_path 
        
    def save(self, loss, epoch, model_state_dict, optimizer_state_dict, history, latest):
        state = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "history": history
        }

        if loss < self.best_loss:
            self.best_loss = loss
            state["best_loss"] = self.best_loss
            torch.save(state, self.save_best_path)
        if latest is True:
            torch.save(state, self.save_latest_path)
