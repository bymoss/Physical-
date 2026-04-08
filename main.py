import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import cv2
from torch.nn.functional import dropout
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

CFG = {
    "BATCH_SIZE": 8,
    "IMG_SIZE": 224,
    "EPOCHS": 9,
    "LR": 1e-4,
    "WEIGHT_DECAY": 5e-2,
    "PATIENCE": 2,
    "MIN_DELTA": 5e-4,
    "BACKBONE": "convnext_tiny.in12k_ft_in1k",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 42,
    "INPUT_MODE": "rgb_object",
    "TRAIN_PATH": "./data/train.csv",
    "VAL_PATH": "./data/dev.csv",
    "DATA_ROOT": "./data",
    "PROCESSED_ROOT": "./processed"
}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 🌟 [추가] 알려주신 평가 산식을 그대로 가져왔습니다.
def get_logloss(true, pred):
    true = np.asarray(true, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    pred = np.nan_to_num(pred, nan=1e-15, posinf=1 - 1e-15, neginf=1e-15)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    loss = -(true * np.log(pred) + (1 - true) * np.log(1 - pred))
    return np.mean(loss)

def freeze_backbone(model):
    for p in model.rgb_backbone.parameters():
        p.requires_grad = False
    if hasattr(model, "edge_backbone"):
        for p in model.edge_backbone.parameters():
            p.requires_grad = False

def unfreeze_backbone(model):
    for p in model.rgb_backbone.parameters():
        p.requires_grad = True
    if hasattr(model, "edge_backbone"):
        for p in model.edge_backbone.parameters():
            p.requires_grad = True

def get_train_transform(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-10, 10),
            shear=(-5, 5),
            p=0.7
        ),
        A.Perspective(scale=(0.02,0.05),p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.12),
            hole_width_range=(0.05, 0.12),
            fill=0,
            p=0.3
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transform(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_edge_transform(img_size=384):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def read_rgb(path: str):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    return img

def debug_batch_shapes(loader):
    batch = next(iter(loader))
    print("=" * 70)
    print("[DEBUG BATCH SHAPES]")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
    print("=" * 70)

class MultiViewDataset(Dataset):
    def __init__(
        self,
        df,
        split,
        data_root,
        processed_root,
        input_mode="rgb",
        rgb_transform=None,
        edge_transform=None,
        is_test=False
    ):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.data_root = Path(data_root)
        self.processed_root = Path(processed_root)
        self.input_mode = input_mode
        self.rgb_transform = rgb_transform
        self.edge_transform = edge_transform
        self.is_test = is_test
        self.label_map = {"stable": 0.0, "unstable": 1.0}

        valid_modes = {"rgb", "object", "rgb_sobel", "object_sobel", "rgb_object"}
        if self.input_mode not in valid_modes:
            raise ValueError(f"지원하지 않는 input_mode: {self.input_mode}")

    def __len__(self):
        return len(self.df)

    def _orig_path(self, sample_id, view):
        return str(self.data_root / self.split / sample_id / f"{view}.png")

    def _obj_path(self, sample_id, view):
        return str(self.processed_root / "object_whitebg" / self.split / sample_id / f"{view}.png")

    def _sobel_path(self, sample_id, view):
        return str(self.processed_root / "edge_sobel_original" / self.split / sample_id / f"{view}.png")

    def _apply_rgb(self, img):
        return self.rgb_transform(image=img)["image"]

    def _apply_edge(self, img):
        x = self.edge_transform(image=img)["image"]
        return x.unsqueeze(0) if x.ndim == 2 else x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        sample = {}

        # 1) 원본 RGB만
        if self.input_mode == "rgb":
            front_rgb = read_rgb(self._orig_path(sample_id, "front"))
            top_rgb = read_rgb(self._orig_path(sample_id, "top"))

            sample["front_rgb"] = self._apply_rgb(front_rgb)
            sample["top_rgb"] = self._apply_rgb(top_rgb)

        # 2) 배경 제거 RGB만
        elif self.input_mode == "object":
            front_obj = read_rgb(self._obj_path(sample_id, "front"))
            top_obj = read_rgb(self._obj_path(sample_id, "top"))

            sample["front_rgb"] = self._apply_rgb(front_obj)
            sample["top_rgb"] = self._apply_rgb(top_obj)

        # 3) 원본 RGB + Sobel
        elif self.input_mode == "rgb_sobel":
            front_rgb = read_rgb(self._orig_path(sample_id, "front"))
            top_rgb = read_rgb(self._orig_path(sample_id, "top"))
            front_edge = read_gray(self._sobel_path(sample_id, "front"))
            top_edge = read_gray(self._sobel_path(sample_id, "top"))

            sample["front_rgb"] = self._apply_rgb(front_rgb)
            sample["top_rgb"] = self._apply_rgb(top_rgb)
            sample["front_edge"] = self._apply_edge(front_edge)
            sample["top_edge"] = self._apply_edge(top_edge)

        # 4) 배경 제거 RGB + Sobel
        elif self.input_mode == "object_sobel":
            front_obj = read_rgb(self._obj_path(sample_id, "front"))
            top_obj = read_rgb(self._obj_path(sample_id, "top"))
            front_edge = read_gray(self._sobel_path(sample_id, "front"))
            top_edge = read_gray(self._sobel_path(sample_id, "top"))

            sample["front_rgb"] = self._apply_rgb(front_obj)
            sample["top_rgb"] = self._apply_rgb(top_obj)
            sample["front_edge"] = self._apply_edge(front_edge)
            sample["top_edge"] = self._apply_edge(top_edge)

        # 5) 원본 RGB + 배경 제거 RGB
        elif self.input_mode == "rgb_object":
            front_rgb = read_rgb(self._orig_path(sample_id, "front"))
            top_rgb = read_rgb(self._orig_path(sample_id, "top"))
            front_obj = read_rgb(self._obj_path(sample_id, "front"))
            top_obj = read_rgb(self._obj_path(sample_id, "top"))

            sample["front_rgb"] = self._apply_rgb(front_rgb)
            sample["top_rgb"] = self._apply_rgb(top_rgb)
            sample["front_obj"] = self._apply_rgb(front_obj)
            sample["top_obj"] = self._apply_rgb(top_obj)

        if not self.is_test:
            label = row["label"]
            label_val = self.label_map[label] if isinstance(label, str) else label
            sample["label"] = torch.tensor([float(label_val)], dtype=torch.float32)

        return sample


class TimmBackbone(nn.Module):
    def __init__(self, model_name, in_chans=3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans,
                                          global_pool="avg")
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone(x)


class MultiInputNet(nn.Module):
    def __init__(self, backbone_name, input_mode="rgb", pretrained=True, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.input_mode = input_mode

        self.use_edge = input_mode in {"rgb_sobel", "object_sobel"}
        self.use_rgb_object = input_mode == "rgb_object"

        # RGB backbone은 공유해서 사용
        self.rgb_backbone = TimmBackbone(backbone_name, in_chans=3, pretrained=pretrained)
        rgb_dim = self.rgb_backbone.num_features

        # Sobel branch
        if self.use_edge:
            self.edge_backbone = TimmBackbone(backbone_name, in_chans=1, pretrained=pretrained)
            edge_dim = self.edge_backbone.num_features
        else:
            edge_dim = 0

        # view 당 feature 차원 계산
        if self.use_rgb_object:
            # 원본 RGB + 배경 제거 RGB
            per_view_dim = rgb_dim * 2
        elif self.use_edge:
            # RGB + Sobel
            per_view_dim = rgb_dim + edge_dim
        else:
            # RGB 하나만
            per_view_dim = rgb_dim

        fusion_dim = per_view_dim * 2  # front + top

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        # rgb_object: 원본 + 배경제거
        if self.use_rgb_object:
            front_rgb_feat = self.rgb_backbone(batch["front_rgb"])
            top_rgb_feat = self.rgb_backbone(batch["top_rgb"])
            front_obj_feat = self.rgb_backbone(batch["front_obj"])
            top_obj_feat = self.rgb_backbone(batch["top_obj"])

            front_feat = torch.cat([front_rgb_feat, front_obj_feat], dim=1)
            top_feat = torch.cat([top_rgb_feat, top_obj_feat], dim=1)

        # rgb_sobel / object_sobel
        elif self.use_edge:
            front_rgb_feat = self.rgb_backbone(batch["front_rgb"])
            top_rgb_feat = self.rgb_backbone(batch["top_rgb"])
            front_edge_feat = self.edge_backbone(batch["front_edge"])
            top_edge_feat = self.edge_backbone(batch["top_edge"])

            front_feat = torch.cat([front_rgb_feat, front_edge_feat], dim=1)
            top_feat = torch.cat([top_rgb_feat, top_edge_feat], dim=1)

        # rgb / object
        else:
            front_feat = self.rgb_backbone(batch["front_rgb"])
            top_feat = self.rgb_backbone(batch["top_rgb"])

        feat = torch.cat([front_feat, top_feat], dim=1)
        return self.classifier(feat)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path =''
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_path = f"./model/best_model{val_loss:.6f}.pt"
            torch.save(model.state_dict(), self.save_path)
            print(f"  --> [SAVE BEST] 최고 기록 갱신! 모델 저장됨 (점수: {val_loss:.6f})")
        else:
            self.counter += 1
            print(f"  --> [EARLY STOP CHECK] 성적이 오르지 않음 ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0


    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Train", leave=False, ncols=100)
    for batch in pbar:
        label = batch["label"].to(device)
        input_batch = {k: v.to(device) for k, v in batch.items() if k != "label"}

        optimizer.zero_grad()
        logits = model(input_batch)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()


        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = label.detach().cpu().numpy()

        all_preds.extend(probs)
        all_labels.extend(labels_np)

        total_loss += loss.item() * label.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")


    custom_logloss_score = get_logloss(all_labels, all_preds)

    return total_loss / len(loader.dataset), custom_logloss_score


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0


    all_labels = []
    all_preds = []

    pbar = tqdm(loader, desc="Valid", leave=False, ncols=100)
    for batch in pbar:
        label = batch["label"].to(device)
        input_batch = {k: v.to(device) for k, v in batch.items() if k != "label"}

        logits = model(input_batch)
        loss = criterion(logits, label)


        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = label.detach().cpu().numpy()

        all_preds.extend(probs)
        all_labels.extend(labels_np)

        total_loss += loss.item() * label.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")


    custom_logloss_score = get_logloss(all_labels, all_preds)

    return total_loss / len(loader.dataset), custom_logloss_score


def main():
    seed_everything(CFG["SEED"])

    train_df = pd.read_csv(CFG["TRAIN_PATH"])
    valid_df = pd.read_csv(CFG["VAL_PATH"])

    print(f"[INFO] 학습 환경: {CFG['DEVICE']}")
    print(f"[INFO] 입력 모드: {CFG['INPUT_MODE']}")
    print(f"[INFO] 훈련 데이터: {len(train_df)}개 | 검증 데이터: {len(valid_df)}개")

    train_dataset = MultiViewDataset(
        df=train_df, split="train",
        data_root=CFG["DATA_ROOT"], processed_root=CFG["PROCESSED_ROOT"],
        input_mode=CFG["INPUT_MODE"],
        rgb_transform=get_train_transform(CFG["IMG_SIZE"]),
        edge_transform=get_edge_transform(CFG["IMG_SIZE"])
    )

    valid_dataset = MultiViewDataset(
        df=valid_df, split="dev",
        data_root=CFG["DATA_ROOT"], processed_root=CFG["PROCESSED_ROOT"],
        input_mode=CFG["INPUT_MODE"],
        rgb_transform=get_valid_transform(CFG["IMG_SIZE"]),
        edge_transform=get_edge_transform(CFG["IMG_SIZE"])
    )

    train_loader = DataLoader(train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False, num_workers=4,
                              pin_memory=True)
    debug_batch_shapes(train_loader)
    model = MultiInputNet(
        backbone_name=CFG["BACKBONE"], input_mode=CFG["INPUT_MODE"], pretrained=True,
    hidden_dim = 128, dropout = 0.5).to(CFG["DEVICE"])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])

    early_stopper = EarlyStopping(patience=CFG["PATIENCE"], min_delta=CFG["MIN_DELTA"])
    freeze_backbone(model)
    for epoch in range(CFG["EPOCHS"]):
        print("\n" + "=" * 60)
        print(f"🏆 [EPOCH {epoch + 1}/{CFG['EPOCHS']}]")


        train_loss, train_logloss = train_one_epoch(model, train_loader, optimizer, criterion, CFG["DEVICE"])
        valid_loss, valid_logloss = valid_one_epoch(model, valid_loader, criterion, CFG["DEVICE"])
        print()
        print(f"[결과] Train LogLoss: {train_logloss:.4f} | Valid LogLoss: {valid_logloss:.4f}")


        early_stopper.step(valid_logloss, model)
        if early_stopper.early_stop:
            print("🚨 [STOP] 더 이상 성능이 개선되지 않아 학습을 조기 종료합니다.")
            break
        # if epoch == 5:
        #     print("[INFO] backbone unfreeze")
        #     unfreeze_backbone(model)
        #     optimizer = torch.optim.AdamW(
        #         filter(lambda p: p.requires_grad, model.parameters()),
        #         lr=5e-6,
        #         weight_decay=CFG["WEIGHT_DECAY"]
        #     )
if __name__ == "__main__":
    main()