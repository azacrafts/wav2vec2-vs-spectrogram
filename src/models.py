import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from torchaudio.transforms import FrequencyMasking, TimeMasking
from transformers import Wav2Vec2Model

from data import SR, LogMel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

BACKBONE_LR = {"cnn": 1e-3, "resnet": 1e-4, "wav2vec": 3e-5}
HEAD_LR = 1e-3
BATCH_SIZE = {"cnn": 32, "resnet": 32, "wav2vec": 16}
EPOCHS = {
    ("cnn", "esc10"): 20,
    ("resnet", "esc10"): 15,
    ("wav2vec", "esc10"): 10,
    ("cnn", "sc"): 10,
    ("resnet", "sc"): 8,
    ("wav2vec", "sc"): 4,
    ("cnn", "sc_small"): 20,
    ("resnet", "sc_small"): 15,
    ("wav2vec", "sc_small"): 10,
}


# random forest
mfcc = torchaudio.transforms.MFCC(SR, n_mfcc=20, melkwargs={"n_fft": 1024, "hop_length": 160, "n_mels": 64})


# function to turn clips into MFCC features
def mfcc_features(X, batch_size=256):
    rows = []
    for i in range(0, len(X), batch_size):
        m = mfcc(torch.from_numpy(X[i : i + batch_size]))
        d1 = torchaudio.functional.compute_deltas(m)
        d2 = torchaudio.functional.compute_deltas(d1)
        coefs = torch.cat([m, d1, d2], dim=1)
        rows.append(torch.cat([coefs.mean(2), coefs.std(2)], dim=1))
    return torch.cat(rows).numpy()


# train RF
def train_rf(train, val, dataset, seed, epochs=None):
    X_train, y_train = train
    start = time.time()
    forest = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    forest.fit(mfcc_features(X_train), y_train)
    train_seconds = time.time() - start

    def predict(X, cpu=False):
        return np.log(forest.predict_proba(mfcc_features(X)) + 1e-6)

    return predict, {"params": 0, "train_seconds": train_seconds}


# CNN
def cnn_body():
    layers, in_channels = [], 1
    for out_channels in (32, 64, 128, 128):
        layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2)]
        in_channels = out_channels
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())


# ResNet18 using already existing resnet18 model from torchvision
def resnet_body():
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    return resnet


# waveform to log-mel to cnn or resnet
class SpecBackbone(nn.Module):

    def __init__(self, body, channels):
        super().__init__()
        self.logmel = LogMel()
        self.freq_mask = FrequencyMasking(10)
        self.time_mask = TimeMasking(20)
        self.body = body
        self.channels = channels

    def forward(self, wav):
        x = self.logmel(wav)
        x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-5)
        if self.training:
            x = self.time_mask(self.freq_mask(x))
        return self.body(x.expand(-1, self.channels, -1, -1))


# wav2vec2
class Wav2VecBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.freeze_feature_encoder()

    def forward(self, wav):
        wav = (wav - wav.mean(1, keepdim=True)) / (wav.std(1, keepdim=True) + 1e-7)
        return self.model(wav).last_hidden_state.mean(1)


# backbone with a classification head
class Net(nn.Module):
    def __init__(self, backbone, dim, n_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_classes))

    def forward(self, wav):
        return self.head(self.backbone(wav))


# fNetwork builder
def build_net(name):
    if name == "cnn":
        return Net(SpecBackbone(cnn_body(), 1), 128)
    if name == "resnet":
        return Net(SpecBackbone(resnet_body(), 3), 512)
    return Net(Wav2VecBackbone(), 768)


# prediction func
def predict_net(model, X, device, batch_size=32):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outputs.append(model(torch.from_numpy(X[i : i + batch_size]).to(device)).cpu())
    return torch.cat(outputs).numpy()


# train any network, not RF
def train_net(name, train, val, dataset, seed, epochs=None):
    X_train, y_train = train
    X_val, y_val = val
    torch.manual_seed(seed)
    epochs = epochs or EPOCHS[(name, dataset)]

    model = build_net(name).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR[name]},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE[name],
        shuffle=True,
        drop_last=True,
    )

    best_loss, best_state = float("inf"), None
    start = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_logits = predict_net(model, X_val, DEVICE)
        val_loss = F.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(y_val)).item()
        print(f"    epoch {epoch + 1}/{epochs}  val_loss {val_loss:.3f}  {time.time() - start:.0f}s", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - start
    model.load_state_dict(best_state)

    def predict(X, cpu=False):
        device = "cpu" if cpu else DEVICE
        model.to(device)
        return predict_net(model, X, device)

    return predict, {"params": sum(p.numel() for p in model.parameters()), "train_seconds": train_seconds}


# train any model
def train_model(name, train, val, dataset, seed, epochs=None):
    if name == "rf":
        return train_rf(train, val, dataset, seed)
    return train_net(name, train, val, dataset, seed, epochs)
