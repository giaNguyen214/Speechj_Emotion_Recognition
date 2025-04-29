
import timm
import xgboost as xgb
import numpy as np
import librosa
import cv2
import torch.nn.functional as F
import math
import json
import pickle
import torch
from torch import nn
import streamlit as st

import os
import gdown

def download_if_not_exists(filepath, gdrive_id):
    if not os.path.exists(filepath):
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        gdown.download(url, filepath, quiet=False)


class Config:

    seed = 42
    sample_rate = 32000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    FS = 32000
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_DURATION = 5
    TARGET_SHAPE = (224, 224)

cfg = Config()

class ConformerXLFeatureExtractor(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):

        super().__init__()
        self.backbone = timm.create_model('caformer_b36.sail_in22k', pretrained=False, num_classes=0)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        features = self.dropout(features)
        if return_features:
            return features
        return self.classifier(features)

def load_model(model_path, model_name):
    checkpoint = torch.load(model_path, map_location=cfg.device)
    if model_name == 'CREMA-D':
        model = ConformerXLFeatureExtractor(num_classes=6)
    else:
        model = ConformerXLFeatureExtractor(num_classes=8)
    # Nếu checkpoint là state_dict trực tiếp:
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(cfg.device)
    model.eval()
    return model


def load_xgb_model(xgb_path):
    with open(xgb_path, 'rb') as f:
        booster = pickle.load(f)
    return booster

def predict_audio(audio_file, model, booster, label_mapping):
    try:
        y, _ = librosa.load(audio_file, sr=cfg.FS)
        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(y) < target_samples:
            y = np.tile(y, int(np.ceil(target_samples / len(y))))
        y = y[:target_samples]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=cfg.FS, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX, power=2.0)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_resized = cv2.resize(mel_db_norm, cfg.TARGET_SHAPE)

        mel_tensor = torch.tensor(mel_resized).unsqueeze(0).unsqueeze(0).float().to(cfg.device)
        mel_tensor = mel_tensor.repeat(1, 3, 1, 1)  # chuyển grayscale -> RGB 3-channel

        with torch.no_grad():
            features = model(mel_tensor, return_features=True)
            features_np = features.cpu().numpy()

        # Predict bằng XGBoost
        pred_probs = booster.predict_proba(features_np)[0]
        probs_dict = {label_mapping[i]: float(pred_probs[i]) for i in range(len(pred_probs))}
        predicted_label = max(probs_dict, key=probs_dict.get)
        return probs_dict, predicted_label

    except Exception as e:
        print(f"Lỗi xử lý file âm thanh: {e}")
        return None, None

def inference(audio_file, model_name):
    if model_name == "CREMA-D":
        # GDrive IDs (bạn cần thay thế bằng ID thật của bạn)
        model_path = 'model/CAFormer.pth'
        xgb_path = 'model/XGBoost.pkl'
        label_path = 'model/label2idx.json'

        download_if_not_exists(model_path, '1H3oy7UI3JIyPkqjeOnfzUWUmo0qVlUeG')
        download_if_not_exists(xgb_path, '13bh0cuWgmEvmNtDBVGd2A3vNhtKV-_2C')
        download_if_not_exists(label_path, '1NFBqE4pzmVi7-B49b_EQW0Jcs3XjfZgd')

        model = load_model(model_path, 'CREMA-D')
        st.write("Successfully loaded CAFormer model - CREMA-D")
        booster = load_xgb_model(xgb_path)
        st.write("Successfully loaded xgb model - CREMA-D")
        with open(label_path, 'r') as f:
            label2idx = json.load(f)
        idx2label = {v: k for k, v in label2idx.items()}
        
        return predict_audio(audio_file, model, booster, idx2label)

    else:
        model_path = 'model/RAVDESS/RAVDESS_saved_model.pth'
        xgb_path = 'model/RAVDESS/RAVDESS_xgboost_model.pkl'
        label_path = 'model/RAVDESS/RAVDESS_label2idx.json'

        download_if_not_exists(model_path, '1OWIYXKkvveWT-o7eVpBYgZkSILIXMaur')
        download_if_not_exists(xgb_path, '1taJAD7dd9ufUD-8HaKTbIiRiJb2rBw9E')
        download_if_not_exists(label_path, '1Sxh2CNnklz7RSNgP9D6jzCf0rAQFCMZT')

        model = load_model(model_path, 'RAVDESS')
        st.write("Successfully loaded CAFormer model - RAVDESS")
        booster = load_xgb_model(xgb_path)
        st.write("Successfully loaded xgb model - RAVDESS")
        with open(label_path, 'r') as f:
            label2idx = json.load(f)
        idx2label = {v: k for k, v in label2idx.items()}

        return predict_audio(audio_file, model, booster, idx2label)

