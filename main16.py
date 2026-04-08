import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm
import multiprocessing
import matplotlib.pyplot as plt


# ==========================================
# 1. 설정 클래스 (CFG)
# ==========================================
class CFG:
    TRAIN_CSV: str = "./data/train.csv"
    DEV_CSV: str = "./data/dev.csv"
    TRAIN_RGB_DIR: str = "./data/train"
    DEV_RGB_DIR: str = "./data/dev"
    TRAIN_MASK_DIR: str = "data/object_whitebg/train"
    DEV_MASK_DIR: str = "data/object_whitebg/dev"
    SAVE_DIR: str = "./model_teacher"

    IMG_SIZE: int = 224
    BATCH_SIZE: int = 16
    EPOCHS: int = 10
    LR: float = 3e-5
    WEIGHT_DECAY: float = 1e-4
    NUM_WORKERS: int = 4
    SEED: int = 42
    NUM_CLASSES: int = 2
    IMAGE_EXT: str = "png"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True
    PATIENCE: int = 5


# ==========================================
# 2. 유틸리티 함수 (그래프 및 시드 고정)
# ==========================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def LOGLOSS(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / np.sum(pred, axis=1).reshape(-1, 1)
    loss = -np.sum(true * np.log(pred), axis=1)
    return np.mean(loss)


# ★ 실시간 그래프 갱신 함수 (반드시 main 호출 전에 정의)
def update_plot(history, save_path='live_learning_curve.png'):
    plt.figure(figsize=(12, 5))

    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['dev_loss'], label='Dev Loss', marker='x')
    plt.title('Real-time Loss')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.legend()

    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['dev_acc'], label='Dev Acc', marker='x')
    plt.title('Real-time Accuracy')
    plt.xlabel('Epoch');
    plt.ylabel('Acc');
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==========================================
# 3. 데이터셋 및 모델 클래스
# ==========================================
class QuadInputDataset(Dataset):
    def __init__(self, csv_file, rgb_dir, mask_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self): return len(self.data_info)

    def __getitem__(self, idx):
        folder_name = str(self.data_info.iloc[idx, 0])
        label_text = str(self.data_info.iloc[idx, 1]).strip().lower()
        label = 0 if label_text == 'stable' else 1 if label_text == 'unstable' else int(label_text)

        file_f, file_t = f'front.{CFG.IMAGE_EXT}', f'top.{CFG.IMAGE_EXT}'
        img_f_r = Image.open(os.path.join(self.rgb_dir, folder_name, file_f)).convert('RGB')
        img_t_r = Image.open(os.path.join(self.rgb_dir, folder_name, file_t)).convert('RGB')
        img_f_m = Image.open(os.path.join(self.mask_dir, folder_name, file_f)).convert('RGB')
        img_t_m = Image.open(os.path.join(self.mask_dir, folder_name, file_t)).convert('RGB')

        if self.transform:
            img_f_r, img_t_r = self.transform(img_f_r), self.transform(img_t_r)
            img_f_m, img_t_m = self.transform(img_f_m), self.transform(img_t_m)
        return img_f_r, img_t_r, img_f_m, img_t_m, label


class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.nets = nn.ModuleList([models.efficientnet_b0(weights='DEFAULT') for _ in range(4)])
        in_features = self.nets[0].classifier[1].in_features
        for net in self.nets: net.classifier = nn.Identity()

        # 과적합 방지를 위해 드롭아웃 추가
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features * 4, CFG.NUM_CLASSES)

    def forward(self, x1, x2, x3, x4):
        feats = [net(x) for net, x in zip(self.nets, [x1, x2, x3, x4])]
        combined = torch.cat(feats, dim=1)
        combined = self.dropout(combined)  # 특징들을 섞기 전에 일부를 끕니다.
        return self.classifier(combined)


# ==========================================
# 4. 핵심 실행 함수 (main)
# ==========================================
def main():
    seed_everything(CFG.SEED)
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)

    history = {'train_loss': [], 'dev_loss': [], 'train_acc': [], 'dev_acc': []}
    train_transform = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
        transforms.RandomRotation(degrees=10),  # ±10도 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 밝기/대비 랜덤 변화
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ★ [수정] Dev/Test용: 평가는 정직하게 원본 그대로 진행
    dev_transform = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 데이터 로더에 각각 다른 transform 적용
    train_dataset = QuadInputDataset(CFG.TRAIN_CSV, CFG.TRAIN_RGB_DIR, CFG.TRAIN_MASK_DIR, train_transform)
    dev_dataset = QuadInputDataset(CFG.DEV_CSV, CFG.DEV_RGB_DIR, CFG.DEV_MASK_DIR, dev_transform)

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    dev_loader = DataLoader(dev_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    model = TeacherNet().to(CFG.DEVICE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)

    # ★ [수정] Weight Decay를 1e-4에서 1e-2로 상향 (모델의 암기 억제)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
    best_dev_logloss = float('inf')
    patience_counter = 0

    print(f"🚀 선생님 모델 학습 시작! (기기: {CFG.DEVICE})\n")

    for epoch in range(CFG.EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for x1, x2, x3, x4, y in train_pbar:
            x1, x2, x3, x4, y = x1.to(CFG.DEVICE), x2.to(CFG.DEVICE), x3.to(CFG.DEVICE), x4.to(CFG.DEVICE), y.to(
                CFG.DEVICE)
            optimizer.zero_grad()
            with torch.autocast('cuda', enabled=CFG.USE_AMP):
                out = model(x1, x2, x3, x4)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update()
            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item();
            train_total += y.size(0)
            train_pbar.set_postfix(loss=train_loss / (train_pbar.n + 1))

        model.eval()
        dev_loss, dev_preds, dev_trues, dev_correct, dev_total = 0, [], [], 0, 0
        dev_pbar = tqdm(dev_loader, desc=f"Epoch {epoch + 1} [Valid]", leave=False)
        with torch.no_grad():
            for x1, x2, x3, x4, y in dev_pbar:
                x1, x2, x3, x4, y = x1.to(CFG.DEVICE), x2.to(CFG.DEVICE), x3.to(CFG.DEVICE), x4.to(CFG.DEVICE), y.to(
                    CFG.DEVICE)
                with torch.autocast('cuda', enabled=CFG.USE_AMP):
                    out = model(x1, x2, x3, x4)
                    loss = loss_fn(out, y)
                dev_loss += loss.item()
                dev_preds.extend(torch.softmax(out, 1).cpu().numpy());
                dev_trues.extend(y.cpu().numpy())
                dev_correct += (out.argmax(1) == y).sum().item();
                dev_total += y.size(0)

        tr_l, dv_l = train_loss / len(train_loader), dev_loss / len(dev_loader)
        tr_a, dv_a = train_correct / train_total, dev_correct / dev_total
        dev_logloss = LOGLOSS(np.eye(CFG.NUM_CLASSES)[np.array(dev_trues)], np.array(dev_preds))

        history['train_loss'].append(tr_l);
        history['dev_loss'].append(dv_l)
        history['train_acc'].append(tr_a);
        history['dev_acc'].append(dv_a)

        # 그래프 업데이트 함수 호출
        update_plot(history)

        status = "✅ 개선됨" if dev_logloss < best_dev_logloss else "❌ 정체"
        print(
            f"[{epoch + 1:02d}/{CFG.EPOCHS}] {status} | Loss: (Tr){tr_l:.4f} (Dv){dv_l:.4f} | Acc: (Tr){tr_a:.2%} (Dv){dv_a:.2%} | LogLoss: {dev_logloss:.4f}")

        if dev_logloss < best_dev_logloss:
            best_dev_logloss = dev_logloss;
            patience_counter = 0
            t_date = datetime.now().strftime("%Y-%m-%d")
            os.makedirs(f"./model_teacher/{t_date}", exist_ok=True)
            torch.save(model.state_dict(), f"./model_teacher/{t_date}/teacher_best_loss_{dev_logloss:.4f}.pth")
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                print(f"\n📢 {CFG.PATIENCE}회 동안 개선이 없어 조기 종료합니다.")
                break


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()