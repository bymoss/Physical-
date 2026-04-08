import os
import csv
import cv2
import math
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class CFG:
    DATA_ROOT = "./data"

    RAW_TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    RAW_DEV_DIR   = os.path.join(DATA_ROOT, "dev")

    BG_TRAIN_DIR = os.path.join(DATA_ROOT, "object_whitebg/train")
    BG_DEV_DIR   = os.path.join(DATA_ROOT, "object_whitebg/dev")

    EDGE_TRAIN_DIR = os.path.join(DATA_ROOT, "edge_sobel_original/train")
    EDGE_DEV_DIR   = os.path.join(DATA_ROOT, "edge_sobel_original/dev")

    TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
    DEV_CSV   = os.path.join(DATA_ROOT, "dev.csv")
    SAVE_DIR = "./teacher_checkpoints"

    IMG_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    FRONT_RAW = "front.png"
    TOP_RAW = "top.png"
    FRONT_BG = "front_bg.png"
    TOP_BG = "top_bg.png"
    FRONT_EDGE = "front_edge.png"
    TOP_EDGE = "top_edge.png"
    VIDEO_NAME = "simulation.mp4"

    VIDEO_FEAT_DIM = 6


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def LOGLOSS(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / np.sum(pred, axis=1).reshape(-1, 1)
    loss = -np.sum(true * np.log(pred), axis=1)
    return np.mean(loss)


def label_to_int(x):
    x = str(x).strip().lower()
    if x == "stable":
        return 0
    if x == "unstable":
        return 1
    raise ValueError(f"Unknown label: {x}")


def read_csv_to_dict(csv_path):
    label_map = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip().lower() for name in reader.fieldnames]
        for row in reader:
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            label_map[row["id"]] = label_to_int(row["label"])
    return label_map


def build_samples(split="train"):
    if split == "train":
        raw_dir = CFG.RAW_TRAIN_DIR
        bg_dir = CFG.BG_TRAIN_DIR
        edge_dir = CFG.EDGE_TRAIN_DIR
        csv_path = CFG.TRAIN_CSV
        require_video = True
    else:
        raw_dir = CFG.RAW_DEV_DIR
        bg_dir = CFG.BG_DEV_DIR
        edge_dir = CFG.EDGE_DEV_DIR
        csv_path = CFG.DEV_CSV
        require_video = False

    label_map = read_csv_to_dict(csv_path)
    samples = []

    for sample_id in sorted(os.listdir(raw_dir)):
        raw_sample = os.path.join(raw_dir, sample_id)
        bg_sample = os.path.join(bg_dir, sample_id)
        edge_sample = os.path.join(edge_dir, sample_id)

        if not os.path.isdir(raw_sample):
            continue

        paths = {
            "front_raw": os.path.join(raw_sample, "front.png"),
            "top_raw": os.path.join(raw_sample, "top.png"),
            "front_bg": os.path.join(bg_sample, "front.png"),
            "top_bg": os.path.join(bg_sample, "top.png"),
            "front_edge": os.path.join(edge_sample, "front.png"),
            "top_edge": os.path.join(edge_sample, "top.png"),
            "video": os.path.join(raw_sample, "simulation.mp4"),
        }

        need_files = [
            paths["front_raw"],
            paths["top_raw"],
            paths["front_bg"],
            paths["top_bg"],
            paths["front_edge"],
            paths["top_edge"],
        ]

        if require_video:
            need_files.append(paths["video"])

        missing = [p for p in need_files if not os.path.isfile(p)]
        if len(missing) > 0:
            print("SKIP:", sample_id)
            for m in missing:
                print("   missing:", m)
            continue

        if sample_id not in label_map:
            print("SKIP CSV:", sample_id)
            continue

        samples.append({
            "id": sample_id,
            "label": label_map[sample_id],
            **paths
        })

    return samples

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"no frames in video: {video_path}")

    return frames, fps


def get_frame_at_time(frames, fps, sec):
    idx = int(round(sec * fps))
    idx = max(0, min(idx, len(frames) - 1))
    return frames[idx], idx


def preprocess_frame(frame, size=(224, 224)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return gray


def frame_diff_score(gray_a, gray_b):
    diff = cv2.absdiff(gray_a, gray_b)
    return float(diff.mean())


def get_binary_mask(gray):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = mask.mean() / 255.0
    if white_ratio > 0.5:
        mask = 255 - mask

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_largest_component_stats(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    x = stats[largest_idx, cv2.CC_STAT_LEFT]
    y = stats[largest_idx, cv2.CC_STAT_TOP]
    w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    area = stats[largest_idx, cv2.CC_STAT_AREA]
    cx, cy = centroids[largest_idx]

    return {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "area": int(area),
        "cx": float(cx),
        "cy": float(cy),
    }


def safe_height_drop_ratio(h0, h1):
    if h0 <= 0:
        return 0.0
    return float(max(0.0, (h0 - h1) / h0))


def safe_centroid_shift(c0x, c0y, c1x, c1y, img_w=224, img_h=224):
    dist = math.sqrt((c1x - c0x) ** 2 + (c1y - c0y) ** 2)
    diag = math.sqrt(img_w ** 2 + img_h ** 2)
    return float(dist / diag)


def extract_video_features(video_path):
    frames, fps = read_video_frames(video_path)

    total_sec = len(frames) / fps
    motion_end_sec = min(3.0, total_sec)
    final_sec = max(0.0, total_sec - (1.0 / fps))

    frame_0_raw, idx_0 = get_frame_at_time(frames, fps, 0.0)
    frame_3_raw, idx_3 = get_frame_at_time(frames, fps, motion_end_sec)
    frame_10_raw, idx_10 = get_frame_at_time(frames, fps, final_sec)

    gray_0 = preprocess_frame(frame_0_raw)
    gray_3 = preprocess_frame(frame_3_raw)
    gray_10 = preprocess_frame(frame_10_raw)

    diff_0_10 = frame_diff_score(gray_0, gray_10)

    motion_scores = []
    motion_end_idx = max(1, idx_3)

    prev_gray = preprocess_frame(frames[0])
    for i in range(1, motion_end_idx + 1):
        cur_gray = preprocess_frame(frames[i])
        score = frame_diff_score(prev_gray, cur_gray)
        motion_scores.append(score)
        prev_gray = cur_gray

    if len(motion_scores) == 0:
        motion_0_3_mean = 0.0
        motion_0_3_max = 0.0
        collapse_time_ratio = 1.0
    else:
        motion_0_3_mean = float(np.mean(motion_scores))
        motion_0_3_max = float(np.max(motion_scores))

        threshold = motion_0_3_max * 0.3
        collapse_idx = None
        for i, v in enumerate(motion_scores):
            if v >= threshold:
                collapse_idx = i + 1
                break

        if collapse_idx is None:
            collapse_time_ratio = 1.0
        else:
            collapse_time_ratio = float(collapse_idx / len(motion_scores))

    mask_0 = get_binary_mask(gray_0)
    mask_10 = get_binary_mask(gray_10)

    stat_0 = get_largest_component_stats(mask_0)
    stat_10 = get_largest_component_stats(mask_10)

    if stat_0 is None or stat_10 is None:
        centroid_shift = 0.0
        height_drop_ratio = 0.0
    else:
        centroid_shift = safe_centroid_shift(
            stat_0["cx"], stat_0["cy"],
            stat_10["cx"], stat_10["cy"],
            img_w=224, img_h=224
        )
        height_drop_ratio = safe_height_drop_ratio(stat_0["h"], stat_10["h"])

    feats = np.array([
        diff_0_10,
        motion_0_3_mean,
        motion_0_3_max,
        centroid_shift,
        height_drop_ratio,
        collapse_time_ratio,
    ], dtype=np.float32)

    return feats


class TeacherDataset(Dataset):
    def __init__(self, samples, transform=None, fit_video_stats=False, video_mean=None, video_std=None):
        self.samples = samples
        self.transform = transform

        self.video_feats = []
        for s in self.samples:
            if os.path.isfile(s["video"]):
                feat = extract_video_features(s["video"])
            else:
                feat = np.zeros(CFG.VIDEO_FEAT_DIM, dtype=np.float32)
            self.video_feats.append(feat)

        if len(self.video_feats) == 0:
            self.video_feats = np.zeros((0, CFG.VIDEO_FEAT_DIM), dtype=np.float32)
        else:
            self.video_feats = np.stack(self.video_feats, axis=0)

        if fit_video_stats:
            self.video_mean = self.video_feats.mean(axis=0) if len(self.video_feats) > 0 else np.zeros(CFG.VIDEO_FEAT_DIM, dtype=np.float32)
            self.video_std = self.video_feats.std(axis=0) + 1e-6 if len(self.video_feats) > 0 else np.ones(CFG.VIDEO_FEAT_DIM, dtype=np.float32)
        else:
            self.video_mean = video_mean
            self.video_std = video_std

    def __len__(self):
        return len(self.samples)

    def _merge_teacher_image(self, sample):
        front_raw = Image.open(sample["front_raw"]).convert("RGB")
        front_bg = Image.open(sample["front_bg"]).convert("RGB")
        front_edge = Image.open(sample["front_edge"]).convert("RGB")

        top_raw = Image.open(sample["top_raw"]).convert("RGB")
        top_bg = Image.open(sample["top_bg"]).convert("RGB")
        top_edge = Image.open(sample["top_edge"]).convert("RGB")

        w, h = front_raw.size

        canvas = Image.new("RGB", (w * 3, h * 2))
        canvas.paste(front_raw, (0, 0))
        canvas.paste(front_bg, (w, 0))
        canvas.paste(front_edge, (w * 2, 0))

        canvas.paste(top_raw, (0, h))
        canvas.paste(top_bg, (w, h))
        canvas.paste(top_edge, (w * 2, h))

        return canvas

    def __getitem__(self, idx):
        sample = self.samples[idx]
        merged = self._merge_teacher_image(sample)

        if self.transform is not None:
            merged = self.transform(merged)

        video_feat = (self.video_feats[idx] - self.video_mean) / self.video_std
        video_feat = torch.tensor(video_feat, dtype=torch.float32)

        label = torch.tensor(sample["label"], dtype=torch.long)
        return merged, video_feat, label


class TeacherNet(nn.Module):
    def __init__(self, video_feat_dim=6):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.video_mlp = nn.Sequential(
            nn.Linear(video_feat_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image, video_feat):
        img_feat = self.backbone(image)
        vid_feat = self.video_mlp(video_feat)
        fused = torch.cat([img_feat, vid_feat], dim=1)
        return self.head(fused)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_labels = []

    for image, video_feat, label in tqdm(loader, desc="Evaluating"):
        image = image.to(CFG.DEVICE)
        video_feat = video_feat.to(CFG.DEVICE)
        label = label.to(CFG.DEVICE)

        logits = model(image, video_feat)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(label.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    onehot = np.eye(2)[all_labels]
    logloss = LOGLOSS(onehot, all_probs)
    return logloss


def train():
    seed_everything()
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    train_samples = build_samples("train")
    dev_samples = build_samples("dev")

    print("Train:", len(train_samples), "Dev:", len(dev_samples))

    train_dataset = TeacherDataset(
        train_samples,
        transform=train_tf,
        fit_video_stats=True
    )

    dev_dataset = TeacherDataset(
        dev_samples,
        transform=eval_tf,
        fit_video_stats=False,
        video_mean=train_dataset.video_mean,
        video_std=train_dataset.video_std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS
    )

    model = TeacherNet(video_feat_dim=CFG.VIDEO_FEAT_DIM).to(CFG.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG.LR,
        weight_decay=CFG.WEIGHT_DECAY
    )

    best_logloss = float("inf")
    patience_count = 0

    for epoch in range(CFG.EPOCHS):
        model.train()

        train_probs = []
        train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CFG.EPOCHS}")

        for image, video_feat, label in pbar:
            image = image.to(CFG.DEVICE)
            video_feat = video_feat.to(CFG.DEVICE)
            label = label.to(CFG.DEVICE)

            optimizer.zero_grad()
            logits = model(image, video_feat)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()  # ✅ 여기서 정상 사용

            pbar.set_postfix(loss=f"{loss_val:.4f}")

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            train_probs.append(probs)
            train_labels.append(label.detach().cpu().numpy())

        train_probs = np.concatenate(train_probs, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        train_onehot = np.eye(2)[train_labels]
        train_logloss = LOGLOSS(train_onehot, train_probs)

        dev_logloss = evaluate(model, dev_loader)

        print(f"{epoch + 1}/{CFG.EPOCHS}")
        print(f"Train LogLoss: {train_logloss:.4f}")
        print(f"Dev   LogLoss: {dev_logloss:.4f}")

        torch.save(model.state_dict(), os.path.join(CFG.SAVE_DIR, "last_teacher.pth"))

        if dev_logloss < best_logloss:
            best_logloss = dev_logloss
            patience_count = 0
            torch.save(model.state_dict(), os.path.join(CFG.SAVE_DIR, "best_teacher.pth"))
            print("✅ best_teacher 저장\n")
        else:
            patience_count += 1
            print(f"patience: {patience_count}/{CFG.PATIENCE}\n")

            if patience_count >= CFG.PATIENCE:
                print("⛔ Early Stopping")
                break


if __name__ == "__main__":
    train()