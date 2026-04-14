# ============================================================
# structure_stability_cv_seed_temp.py
# - GPU 3번 사용
# - train + dev 전체를 Stratified 5-Fold로 학습
# - seed ensemble 지원
# - handcrafted feature fold별 표준화(mean/std)
# - label smoothing 적용
# - fold별 temperature scaling 적용
# - 최종 test는 (seed x fold) 전체 평균 앙상블
# - 제출 형식: id, unstable_prob, stable_prob
# ============================================================

import os

# ---------------- GPU 3번 고정 ----------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import cv2
import math
import copy
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# ============================================================
# 0. CONFIG
# ============================================================
class CFG:
    # ---------- 경로 ----------
    ROOT_DIR = "./young_data/data"

    TRAIN_DIR = os.path.join(ROOT_DIR, "train")
    DEV_DIR = os.path.join(ROOT_DIR, "dev")
    TEST_DIR = os.path.join(ROOT_DIR, "test")

    TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
    DEV_CSV = os.path.join(ROOT_DIR, "dev.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(ROOT_DIR, "sample_submission.csv")

    # ---------- 파일명 ----------
    FRONT_IMG_NAME = "front.png"
    TOP_IMG_NAME = "top.png"

    # ---------- CSV 컬럼명 ----------
    ID_COL = "id"
    LABEL_COL = "label"

    # 반드시 0=unstable, 1=stable
    LABEL_MAP = {
        "unstable": 0,
        "stable": 1,
    }

    # ---------- 학습 ----------
    SEEDS = [42, 2024]          # seed ensemble
    N_SPLITS = 5

    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4

    USE_PRETRAINED = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    USE_WEIGHTED_SAMPLER = False
    PATIENCE = 5

    # ---------- loss ----------
    LABEL_SMOOTHING = 0.03

    # ---------- temperature scaling ----------
    USE_TEMPERATURE_SCALING = True
    TEMP_GRID = [
        0.50, 0.60, 0.70, 0.80, 0.90,
        1.00, 1.10, 1.20, 1.30, 1.40,
        1.50, 1.60, 1.70, 1.80, 2.00,
        2.20, 2.50, 3.00
    ]

    # ---------- 저장 ----------
    SAVE_DIR = "./young_data/outputs_seed_temp"
    SUBMISSION_NAME = "submission_seed_temp.csv"
    OOF_NAME = "oof_predictions_seed_temp.csv"
    MODEL_DIR = os.path.join(SAVE_DIR, "models")
    TEMP_DIR = os.path.join(SAVE_DIR, "temps")
    FEATSTAT_DIR = os.path.join(SAVE_DIR, "feat_stats")

    # ---------- 추론 ----------
    USE_TTA = True
    TTA_TIMES = 4

    # ---------- 실행 모드 ----------
    RUN_CV_TRAIN = True
    RUN_CV_PREDICT = True


os.makedirs(CFG.SAVE_DIR, exist_ok=True)
os.makedirs(CFG.MODEL_DIR, exist_ok=True)
os.makedirs(CFG.TEMP_DIR, exist_ok=True)
os.makedirs(CFG.FEATSTAT_DIR, exist_ok=True)


# ============================================================
# 1. UTIL
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    print(msg, flush=True)


def safe_read_image(img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def normalize_label(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x in CFG.LABEL_MAP:
            return CFG.LABEL_MAP[x]
    return int(x)


def get_model_path(seed: int, fold: int):
    return os.path.join(CFG.MODEL_DIR, f"best_seed{seed}_fold{fold}.pt")


def get_feat_mean_path(seed: int, fold: int):
    return os.path.join(CFG.FEATSTAT_DIR, f"seed{seed}_fold{fold}_feat_mean.npy")


def get_feat_std_path(seed: int, fold: int):
    return os.path.join(CFG.FEATSTAT_DIR, f"seed{seed}_fold{fold}_feat_std.npy")


def get_temp_path(seed: int, fold: int):
    return os.path.join(CFG.TEMP_DIR, f"seed{seed}_fold{fold}_temperature.npy")


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================
def make_foreground_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    체커보드 배경 + 컬러 블록 구조물에서 foreground 분리용 heuristic mask
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    mask_sat = (sat > 45).astype(np.uint8) * 255

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)

    mask = cv2.bitwise_or(mask_sat, edges)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)
    fg = (labels == largest_idx).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fg


def contour_and_hull(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    return cnt, hull


def mask_centroid(mask: np.ndarray):
    m = cv2.moments(mask)
    h, w = mask.shape[:2]
    if m["m00"] == 0:
        return w / 2, h / 2
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cx, cy


def bounding_box_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, w - 1, h - 1
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return x1, y1, x2, y2


def extract_handcrafted_features(front_img: np.ndarray, top_img: np.ndarray) -> np.ndarray:
    front_mask = make_foreground_mask(front_img)
    top_mask = make_foreground_mask(top_img)

    fh, fw = front_mask.shape
    th, tw = top_mask.shape

    f_cnt, f_hull = contour_and_hull(front_mask)
    t_cnt, t_hull = contour_and_hull(top_mask)

    f_cx, f_cy = mask_centroid(front_mask)
    t_cx, t_cy = mask_centroid(top_mask)

    fx1, fy1, fx2, fy2 = bounding_box_from_mask(front_mask)
    tx1, ty1, tx2, ty2 = bounding_box_from_mask(top_mask)

    f_bw = max(1, fx2 - fx1 + 1)
    f_bh = max(1, fy2 - fy1 + 1)
    t_bw = max(1, tx2 - tx1 + 1)
    t_bh = max(1, ty2 - ty1 + 1)

    f_area = float((front_mask > 0).sum())
    t_area = float((top_mask > 0).sum())

    f_hull_area = float(cv2.contourArea(f_hull)) if f_hull is not None else 0.0
    t_hull_area = float(cv2.contourArea(t_hull)) if t_hull is not None else 0.0

    f_peri = float(cv2.arcLength(f_cnt, True)) if f_cnt is not None else 0.0
    t_peri = float(cv2.arcLength(t_cnt, True)) if t_cnt is not None else 0.0

    f_bbox_area = float(f_bw * f_bh)
    t_bbox_area = float(t_bw * t_bh)

    f_fill = f_area / (f_bbox_area + 1e-6)
    t_fill = t_area / (t_bbox_area + 1e-6)

    f_compact = (4.0 * math.pi * f_area) / (f_peri * f_peri + 1e-6)
    t_compact = (4.0 * math.pi * t_area) / (t_peri * t_peri + 1e-6)

    f_cx_n = f_cx / fw
    f_cy_n = f_cy / fh
    t_cx_n = t_cx / tw
    t_cy_n = t_cy / th

    f_bcx_n = ((fx1 + fx2) * 0.5) / fw
    f_bcy_n = ((fy1 + fy2) * 0.5) / fh
    t_bcx_n = ((tx1 + tx2) * 0.5) / tw
    t_bcy_n = ((ty1 + ty2) * 0.5) / th

    f_center_offset = math.sqrt((f_cx_n - f_bcx_n) ** 2 + (f_cy_n - f_bcy_n) ** 2)
    t_center_offset = math.sqrt((t_cx_n - t_bcx_n) ** 2 + (t_cy_n - t_bcy_n) ** 2)

    f_aspect = f_bh / (f_bw + 1e-6)
    t_aspect = t_bh / (t_bw + 1e-6)

    f_hull_fill = f_area / (f_hull_area + 1e-6)
    t_hull_fill = t_area / (t_hull_area + 1e-6)

    f_width_ratio = f_bw / fw
    f_height_ratio = f_bh / fh
    t_width_ratio = t_bw / tw
    t_height_ratio = t_bh / th

    t_global_center_offset = math.sqrt((t_cx_n - 0.5) ** 2 + (t_cy_n - 0.5) ** 2)

    front_upper = front_mask[: fh // 2, :]
    front_lower = front_mask[fh // 2 :, :]
    upper_area = float((front_upper > 0).sum())
    lower_area = float((front_lower > 0).sum())
    top_heaviness = upper_area / (lower_area + 1e-6)

    front_left = front_mask[:, : fw // 2]
    front_right = front_mask[:, fw // 2 :]
    left_area = float((front_left > 0).sum())
    right_area = float((front_right > 0).sum())
    left_right_balance = abs(left_area - right_area) / (f_area + 1e-6)

    top_left = top_mask[:, : tw // 2]
    top_right = top_mask[:, tw // 2 :]
    t_left_area = float((top_left > 0).sum())
    t_right_area = float((top_right > 0).sum())
    top_lr_balance = abs(t_left_area - t_right_area) / (t_area + 1e-6)

    pseudo_support_score = (
        (t_width_ratio + t_height_ratio) * 0.5
        - 0.7 * t_center_offset
        - 0.5 * t_global_center_offset
    )

    pseudo_risk_score = (
        0.8 * f_aspect
        + 0.8 * top_heaviness
        + 1.2 * t_center_offset
        + 0.8 * left_right_balance
        - 0.7 * pseudo_support_score
    )

    feats = np.array([
        f_cx_n, f_cy_n,
        f_bcx_n, f_bcy_n,
        f_center_offset,
        f_aspect,
        f_fill,
        f_compact,
        f_hull_fill,
        f_width_ratio,
        f_height_ratio,
        top_heaviness,
        left_right_balance,

        t_cx_n, t_cy_n,
        t_bcx_n, t_bcy_n,
        t_center_offset,
        t_aspect,
        t_fill,
        t_compact,
        t_hull_fill,
        t_width_ratio,
        t_height_ratio,
        t_global_center_offset,
        top_lr_balance,

        pseudo_support_score,
        pseudo_risk_score,
        f_area / (fw * fh + 1e-6),
        t_area / (tw * th + 1e-6),
        f_hull_area / (fw * fh + 1e-6),
        t_hull_area / (tw * th + 1e-6),
    ], dtype=np.float32)

    return feats


# ============================================================
# 3. PREPROCESS / AUG
# ============================================================
def resize_img(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def random_augment(img: np.ndarray) -> np.ndarray:
    img = img.copy()

    if random.random() < 0.5:
        img = np.fliplr(img).copy()

    if random.random() < 0.8:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        h, w = img.shape[:2]
        angle = random.uniform(-7, 7)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    if random.random() < 0.2:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    return img


# ============================================================
# 4. DATA PREP
# ============================================================
def load_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if CFG.ID_COL not in df.columns:
        raise ValueError(f"{csv_path} 에 '{CFG.ID_COL}' 컬럼이 없습니다.")
    return df


def build_full_labeled_df():
    train_df = load_df(CFG.TRAIN_CSV).copy()
    dev_df = load_df(CFG.DEV_CSV).copy()

    train_df["_root"] = CFG.TRAIN_DIR
    dev_df["_root"] = CFG.DEV_DIR

    full_df = pd.concat([train_df, dev_df], axis=0).reset_index(drop=True)
    full_df[CFG.LABEL_COL] = full_df[CFG.LABEL_COL].apply(normalize_label)
    return full_df


def build_test_df():
    test_df = pd.read_csv(CFG.SAMPLE_SUBMISSION_CSV).copy()
    if CFG.ID_COL not in test_df.columns:
        raise ValueError(f"{CFG.SAMPLE_SUBMISSION_CSV} 에 '{CFG.ID_COL}' 컬럼이 없습니다.")
    test_df["_root"] = CFG.TEST_DIR
    return test_df


def get_sample_paths(sample_id: str, root_dir: str):
    sample_dir = os.path.join(root_dir, str(sample_id))
    front_path = os.path.join(sample_dir, CFG.FRONT_IMG_NAME)
    top_path = os.path.join(sample_dir, CFG.TOP_IMG_NAME)
    return front_path, top_path


def build_feature_cache(df: pd.DataFrame):
    feature_cache = {}
    for _, row in df.iterrows():
        sample_id = str(row[CFG.ID_COL])
        root_dir = row["_root"]
        cache_key = f"{root_dir}_{sample_id}"

        if cache_key in feature_cache:
            continue

        front_path, top_path = get_sample_paths(sample_id, root_dir)
        front_img = safe_read_image(front_path)
        top_img = safe_read_image(top_path)
        feature_cache[cache_key] = extract_handcrafted_features(front_img, top_img)

    return feature_cache


def compute_feature_stats(df: pd.DataFrame, feature_cache: dict):
    feats = []
    for _, row in df.iterrows():
        sample_id = str(row[CFG.ID_COL])
        root_dir = row["_root"]
        cache_key = f"{root_dir}_{sample_id}"
        feats.append(feature_cache[cache_key])

    feats = np.stack(feats, axis=0).astype(np.float32)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


# ============================================================
# 5. DATASET
# ============================================================
class StructureDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mode: str,
        feature_cache: dict,
        feat_mean: np.ndarray = None,
        feat_std: np.ndarray = None,
    ):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.feature_cache = feature_cache
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = str(row[CFG.ID_COL])
        root_dir = row["_root"]
        cache_key = f"{root_dir}_{sample_id}"

        front_path, top_path = get_sample_paths(sample_id, root_dir)
        front_img = safe_read_image(front_path)
        top_img = safe_read_image(top_path)

        feats = self.feature_cache[cache_key].copy()
        if self.feat_mean is not None and self.feat_std is not None:
            feats = (feats - self.feat_mean) / (self.feat_std + 1e-6)

        if self.mode == "train":
            front_img = random_augment(front_img)
            top_img = random_augment(top_img)

        front_img = resize_img(front_img, CFG.IMG_SIZE)
        top_img = resize_img(top_img, CFG.IMG_SIZE)

        front_tensor = to_tensor(front_img)
        top_tensor = to_tensor(top_img)
        feat_tensor = torch.tensor(feats, dtype=torch.float32)

        if self.mode in ["train", "valid"]:
            label = int(row[CFG.LABEL_COL])
            label_tensor = torch.tensor(label, dtype=torch.long)
            return {
                "front": front_tensor,
                "top": top_tensor,
                "feat": feat_tensor,
                "label": label_tensor,
                "id": sample_id,
            }

        return {
            "front": front_tensor,
            "top": top_tensor,
            "feat": feat_tensor,
            "id": sample_id,
        }


# ============================================================
# 6. MODEL
# ============================================================
class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return x


class StructureNet(nn.Module):
    def __init__(self, feat_dim: int, pretrained=True):
        super().__init__()
        self.encoder = ImageEncoder(pretrained=pretrained)
        img_dim = self.encoder.out_dim

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        fusion_dim = img_dim * 4 + 64

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, front, top, feat):
        f1 = self.encoder(front)
        f2 = self.encoder(top)

        fdiff = torch.abs(f1 - f2)
        fmul = f1 * f2

        hfeat = self.feat_mlp(feat)

        x = torch.cat([f1, f2, fdiff, fmul, hfeat], dim=1)
        out = self.head(x)
        return out


# ============================================================
# 7. METRIC / TEMPERATURE
# ============================================================
def multiclass_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    y_prob = y_prob / np.sum(y_prob, axis=1, keepdims=True)

    y_onehot = np.eye(2)[y_true]
    loss = -np.sum(y_onehot * np.log(y_prob), axis=1)
    return np.mean(loss)


def softmax_np(logits: np.ndarray):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def apply_temperature_to_logits(logits: np.ndarray, temperature: float):
    return logits / float(temperature)


def find_best_temperature(logits: np.ndarray, labels: np.ndarray):
    best_temp = 1.0
    best_score = float("inf")

    for t in CFG.TEMP_GRID:
        scaled_logits = apply_temperature_to_logits(logits, t)
        probs = softmax_np(scaled_logits)
        score = multiclass_logloss(labels, probs)

        if score < best_score:
            best_score = score
            best_temp = t

    return float(best_temp), float(best_score)


# ============================================================
# 8. TRAIN / VALID / INFER
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)
        feat = batch["feat"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(front, top, feat)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_logits = []
    all_ids = []

    for batch in loader:
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)
        feat = batch["feat"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        ids = batch["id"]

        logits = model(front, top, feat)
        loss = criterion(logits, label)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * label.size(0)
        all_labels.append(label.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_ids.extend(ids)

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    logloss = multiclass_logloss(all_labels, all_probs)
    return total_loss / len(loader.dataset), logloss, all_ids, all_labels, all_probs, all_logits


@torch.no_grad()
def inference(model, loader, device, use_tta=False, tta_times=4, temperature=1.0):
    model.eval()
    all_ids = []
    all_probs = []

    for batch in loader:
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)
        feat = batch["feat"].to(device, non_blocking=True)
        ids = batch["id"]

        if not use_tta:
            logits = model(front, top, feat)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=1)
        else:
            probs_sum = 0.0
            for _ in range(tta_times):
                f_in = front.clone()
                t_in = top.clone()

                if random.random() < 0.5:
                    f_in = torch.flip(f_in, dims=[3])
                if random.random() < 0.5:
                    t_in = torch.flip(t_in, dims=[3])

                logits = model(f_in, t_in, feat)
                logits = logits / temperature
                probs_sum += torch.softmax(logits, dim=1)

            probs = probs_sum / tta_times

        all_ids.extend(ids)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    return all_ids, all_probs


# ============================================================
# 9. LOADER HELPERS
# ============================================================
def make_loader(dataset, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )


def make_weighted_sampler(df: pd.DataFrame):
    labels = df[CFG.LABEL_COL].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ============================================================
# 10. CV TRAIN (SEED ENSEMBLE + TEMP SCALING)
# ============================================================
def train_cv():
    full_df = build_full_labeled_df()
    feature_cache = build_feature_cache(full_df)

    feat_dim = len(next(iter(feature_cache.values())))
    y = full_df[CFG.LABEL_COL].values

    oof_probs = np.zeros((len(full_df), 2), dtype=np.float32)
    oof_counts = np.zeros(len(full_df), dtype=np.float32)

    all_fold_scores = []
    all_temp_scores = []

    log(f"Total labeled samples: {len(full_df)}")
    log(f"Feature dim: {feat_dim}")
    log(f"Seeds: {CFG.SEEDS}")

    for seed in CFG.SEEDS:
        log("\n" + "#" * 80)
        log(f"START SEED = {seed}")
        log("#" * 80)

        seed_everything(seed)

        skf = StratifiedKFold(
            n_splits=CFG.N_SPLITS,
            shuffle=True,
            random_state=seed
        )

        for fold, (train_idx, valid_idx) in enumerate(skf.split(full_df, y), start=1):
            log("=" * 80)
            log(f"Seed {seed} | Fold {fold}/{CFG.N_SPLITS}")
            log("=" * 80)

            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            valid_df = full_df.iloc[valid_idx].reset_index(drop=True)

            feat_mean, feat_std = compute_feature_stats(train_df, feature_cache)
            np.save(get_feat_mean_path(seed, fold), feat_mean)
            np.save(get_feat_std_path(seed, fold), feat_std)

            train_dataset = StructureDataset(
                train_df,
                mode="train",
                feature_cache=feature_cache,
                feat_mean=feat_mean,
                feat_std=feat_std,
            )
            valid_dataset = StructureDataset(
                valid_df,
                mode="valid",
                feature_cache=feature_cache,
                feat_mean=feat_mean,
                feat_std=feat_std,
            )

            if CFG.USE_WEIGHTED_SAMPLER:
                train_sampler = make_weighted_sampler(train_df)
                train_loader = make_loader(train_dataset, sampler=train_sampler)
            else:
                train_loader = make_loader(train_dataset, shuffle=True)

            valid_loader = make_loader(valid_dataset, shuffle=False)

            model = StructureNet(
                feat_dim=feat_dim,
                pretrained=CFG.USE_PRETRAINED
            ).to(CFG.DEVICE)

            criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=CFG.LR,
                weight_decay=CFG.WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=CFG.EPOCHS
            )

            best_score = float("inf")
            best_epoch = -1
            best_state = None
            patience_count = 0

            for epoch in range(1, CFG.EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.DEVICE)
                valid_loss, valid_logloss, _, _, _, _ = valid_one_epoch(model, valid_loader, criterion, CFG.DEVICE)
                scheduler.step()

                log(
                    f"[Seed {seed} | Fold {fold} | Epoch {epoch:02d}] "
                    f"Train Loss: {train_loss:.5f} | "
                    f"Valid Loss: {valid_loss:.5f} | "
                    f"Valid LogLoss: {valid_logloss:.5f}"
                )

                if valid_logloss < best_score:
                    best_score = valid_logloss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save(best_state, get_model_path(seed, fold))
                    log("✅ best model saved")
                    patience_count = 0
                else:
                    patience_count += 1
                    log(f"patience: {patience_count}/{CFG.PATIENCE}")

                if patience_count >= CFG.PATIENCE:
                    log("⏹ Early stopping")
                    break

            log(f"Seed {seed} Fold {fold} Best Epoch: {best_epoch}")
            log(f"Seed {seed} Fold {fold} Best Valid LogLoss: {best_score:.6f}")

            # ---------- best model reload ----------
            best_model = StructureNet(
                feat_dim=feat_dim,
                pretrained=False
            ).to(CFG.DEVICE)
            best_model.load_state_dict(torch.load(get_model_path(seed, fold), map_location=CFG.DEVICE))

            # ---------- validation logits ----------
            _, raw_logloss, valid_ids, valid_labels, valid_probs_raw, valid_logits = valid_one_epoch(
                best_model, valid_loader, criterion, CFG.DEVICE
            )

            # ---------- temperature scaling ----------
            if CFG.USE_TEMPERATURE_SCALING:
                best_temp, temp_logloss = find_best_temperature(valid_logits, valid_labels)
            else:
                best_temp, temp_logloss = 1.0, raw_logloss

            np.save(get_temp_path(seed, fold), np.array([best_temp], dtype=np.float32))

            scaled_probs = softmax_np(valid_logits / best_temp)

            log(f"Seed {seed} Fold {fold} Raw LogLoss : {raw_logloss:.6f}")
            log(f"Seed {seed} Fold {fold} Best Temp   : {best_temp:.4f}")
            log(f"Seed {seed} Fold {fold} Temp LogLoss: {temp_logloss:.6f}")

            oof_probs[valid_idx] += scaled_probs
            oof_counts[valid_idx] += 1.0

            all_fold_scores.append(best_score)
            all_temp_scores.append(temp_logloss)

    # 각 샘플은 seed 개수만큼 valid에 들어감
    oof_probs = oof_probs / oof_counts[:, None]
    oof_logloss = multiclass_logloss(y, oof_probs)

    oof_df = full_df[[CFG.ID_COL, CFG.LABEL_COL, "_root"]].copy()
    oof_df["unstable_prob"] = oof_probs[:, 0]
    oof_df["stable_prob"] = oof_probs[:, 1]
    oof_df.to_csv(os.path.join(CFG.SAVE_DIR, CFG.OOF_NAME), index=False)

    log("\n" + "=" * 80)
    log("CV FINISHED")
    log(f"Raw Fold Mean LogLoss : {np.mean(all_fold_scores):.6f}")
    log(f"Temp Fold Mean LogLoss: {np.mean(all_temp_scores):.6f}")
    log(f"Final OOF LogLoss     : {oof_logloss:.6f}")
    log("=" * 80)


# ============================================================
# 11. SUBMISSION SAVE
# ============================================================
def save_submission(ids, probs, save_path):
    sub = pd.DataFrame({
        CFG.ID_COL: ids,
        "unstable_prob": probs[:, 0],
        "stable_prob": probs[:, 1],
    })

    row_sum = sub[["unstable_prob", "stable_prob"]].sum(axis=1).values
    sub["unstable_prob"] = sub["unstable_prob"] / row_sum
    sub["stable_prob"] = sub["stable_prob"] / row_sum

    sub = sub[[CFG.ID_COL, "unstable_prob", "stable_prob"]]
    sub.to_csv(save_path, index=False)
    log(f"✅ submission saved: {save_path}")


# ============================================================
# 12. TEST PREDICT WITH (SEED x FOLD) ENSEMBLE
# ============================================================
def predict_test_cv():
    full_df = build_full_labeled_df()
    test_df = build_test_df()

    full_feature_cache = build_feature_cache(full_df)
    test_feature_cache = build_feature_cache(test_df)

    feat_dim = len(next(iter(full_feature_cache.values())))

    test_ids_ref = None
    test_probs_sum = None
    model_count = 0

    for seed in CFG.SEEDS:
        for fold in range(1, CFG.N_SPLITS + 1):
            log("=" * 80)
            log(f"Test inference | Seed {seed} | Fold {fold}")
            log("=" * 80)

            feat_mean = np.load(get_feat_mean_path(seed, fold))
            feat_std = np.load(get_feat_std_path(seed, fold))
            temperature = float(np.load(get_temp_path(seed, fold))[0])

            test_dataset = StructureDataset(
                test_df,
                mode="test",
                feature_cache=test_feature_cache,
                feat_mean=feat_mean,
                feat_std=feat_std,
            )
            test_loader = make_loader(test_dataset, shuffle=False)

            model = StructureNet(
                feat_dim=feat_dim,
                pretrained=False
            ).to(CFG.DEVICE)

            model.load_state_dict(torch.load(get_model_path(seed, fold), map_location=CFG.DEVICE))
            model.eval()

            ids, probs = inference(
                model,
                test_loader,
                CFG.DEVICE,
                use_tta=CFG.USE_TTA,
                tta_times=CFG.TTA_TIMES,
                temperature=temperature
            )

            if test_ids_ref is None:
                test_ids_ref = ids
                test_probs_sum = probs
            else:
                test_probs_sum += probs

            model_count += 1

    test_probs_avg = test_probs_sum / model_count
    save_path = os.path.join(CFG.SAVE_DIR, CFG.SUBMISSION_NAME)
    save_submission(test_ids_ref, test_probs_avg, save_path)

    log(f"Total ensembled models: {model_count}")


# ============================================================
# 13. MAIN
# ============================================================
if __name__ == "__main__":
    log(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    log(f"Using device: {CFG.DEVICE}")
    if torch.cuda.is_available():
        log(f"Visible CUDA device count: {torch.cuda.device_count()}")
        log(f"Current CUDA device index: {torch.cuda.current_device()}")

    if CFG.RUN_CV_TRAIN:
        train_cv()

    if CFG.RUN_CV_PREDICT:
        predict_test_cv()