import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F


class FaceDecoderLoss(nn.Module):
    def __init__(self, coef_landmarks=1, coef_textures=100, coef_embeddings=100):
        super().__init__()
        self.coef_landmarks = coef_landmarks
        self.coef_textures = coef_textures
        self.coef_embeddings = coef_embeddings
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
            loss_landmarks = self.coef_landmarks * self.mse_loss(landmarks_true, landmarks_predicted)
            sum_loss += loss_landmarks
        # MAE for textures
        if textures_true is not None and textures_predicted is not None:
            loss_textures = self.coef_textures * self.mae_loss(textures_true, textures_predicted)
            sum_loss += loss_textures
        # Cosine Similarity loss for embeddings
        if embeddings_true is not None and embeddings_predicted is not None:
            loss_embeddings = self.coef_embeddings * self.cos_loss(embeddings_true, embeddings_predicted, torch.tensor([1]).to("cuda"))
            sum_loss += loss_embeddings

        return sum_loss, loss_landmarks, loss_textures, loss_embeddings


class S2FLoss(nn.Module):
    def __init__(self, face_encoder_last_layer, face_decoder_first_layer, coe_1, coe_2, coe_3):
        super().__init__()
        self.face_encoder_last_layer = face_encoder_last_layer
        self.face_decoder_first_layer = face_decoder_first_layer
        self.mae_loss = nn.L1Loss()
        self.coe_1 = coe_1
        self.coe_2 = coe_2
        self.coe_3 = coe_3

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
        loss_face_decoder *= self.coe_3
        sum_loss += loss_face_decoder

        return sum_loss, loss_base, loss_face_encoder, loss_face_decoder
    
    def knowledge_distilation(self, a, b, T=2):
        p_a = F.softmax(a / T, dim=1)
        p_b = F.log_softmax(b / T, dim=1)
        return -(p_a * p_b).sum()


# class S2FLoss(nn.Module):
#     def __init__(self, face_encoder_last_layer, face_decoder_first_layer, coe_1, coe_2, coe_3):
#         super().__init__()
#         self.face_encoder_last_layer = face_encoder_last_layer
#         self.face_decoder_first_layer = face_decoder_first_layer
#         self.mae_loss = nn.L1Loss(reduction="none")
#         self.coe_1 = coe_1
#         self.coe_2 = coe_2
#         self.coe_3 = coe_3

#     def forward(self, pred, true):
#         sum_loss = 0
#         loss_base, loss_face_encoder, loss_face_decoder = 0, 0, 0

#         # loss_base - v_f and v_s distance part
#         vs_normalized = pred / norm(pred, dim=1).view(pred.shape[0], 1)
#         vf_normalized = true / norm(true, dim=1).view(true.shape[0], 1)
#         loss_base = torch.pow(norm(vf_normalized - vs_normalized, dim=1), 2).mean()
#         loss_base *= self.coe_1
#         sum_loss += loss_base

#         # loss_face_encoder - face encoder last layer activation part
#         vgg_v_s = self.face_encoder_last_layer(pred)
#         with torch.no_grad():
#             vgg_v_f = self.face_encoder_last_layer(true)
#         loss_face_encoder = self.knowledge_distilation(vgg_v_f, vgg_v_s).mean()
#         loss_face_encoder *= self.coe_2
#         sum_loss += loss_face_encoder

#         # loss_face_decoder - face decoder first layers activation part
#         dec_v_s = self.face_decoder_first_layer(pred)
#         with torch.no_grad():
#             dec_v_f = self.face_decoder_first_layer(true)
#         loss_face_decoder = self.mae_loss(dec_v_f, dec_v_s).sum(dim=1).mean()
#         loss_face_decoder *= self.coe_3
#         sum_loss += loss_face_decoder
        
#         return sum_loss, loss_base, loss_face_encoder, loss_face_decoder
    
#     def knowledge_distilation(self, a, b, T=2):
#         p_a = F.softmax(a / T, dim=1)
#         p_b = F.log_softmax(b / T, dim=1)
#         return -(p_a * p_b).sum(dim=1)
