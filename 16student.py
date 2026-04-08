import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm
import multiprocessing


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

    # ★ [필수 수정] 학습된 선생님 모델(.pth)의 경로를 입력하세요
    TEACHER_MODEL_PATH: str = "./model_teacher/2026-03-28/teacher_best_loss_0.5489.pth"

    SAVE_DIR: str = "./model_student"  # 학생 모델 저장 루트 폴더

    IMG_SIZE: int = 224
    BATCH_SIZE: int = 16
    EPOCHS: int = 30
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-3
    NUM_WORKERS: int = 4
    SEED: int = 42

    # KD 설정
    TEMPERATURE: float = 4.0
    ALPHA: float = 0.5

    NUM_CLASSES: int = 2
    IMAGE_EXT: str = "png"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True
    PATIENCE: int = 7


# ==========================================
# 2. 유틸리티 및 데이터셋 (기존 구조 유지)
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

        f_name, t_name = f'front.{CFG.IMAGE_EXT}', f'top.{CFG.IMAGE_EXT}'
        img_f_r = Image.open(os.path.join(self.rgb_dir, folder_name, f_name)).convert('RGB')
        img_t_r = Image.open(os.path.join(self.rgb_dir, folder_name, t_name)).convert('RGB')
        img_f_m = Image.open(os.path.join(self.mask_dir, folder_name, f_name)).convert('RGB')
        img_t_m = Image.open(os.path.join(self.mask_dir, folder_name, t_name)).convert('RGB')

        if self.transform:
            img_f_r = self.transform(img_f_r);
            img_t_r = self.transform(img_t_r)
            img_f_m = self.transform(img_f_m);
            img_t_m = self.transform(img_t_m)
        return img_f_r, img_t_r, img_f_m, img_t_m, label


# ==========================================
# 3. 모델 아키텍처
# ==========================================
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.nets = nn.ModuleList([models.efficientnet_b0(weights=None) for _ in range(4)])
        in_features = self.nets[0].classifier[1].in_features
        for net in self.nets: net.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features * 4, CFG.NUM_CLASSES)

    def forward(self, x1, x2, x3, x4):
        feats = [net(x) for net, x in zip(self.nets, [x1, x2, x3, x4])]
        return self.classifier(torch.cat(feats, dim=1))


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.net_f = models.efficientnet_b0(weights='DEFAULT')
        self.net_t = models.efficientnet_b0(weights='DEFAULT')
        in_features = self.net_f.classifier[1].in_features
        self.net_f.classifier = nn.Identity()
        self.net_t.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features * 2, CFG.NUM_CLASSES)

    def forward(self, x_f, x_t):
        feat_f = self.net_f(x_f);
        feat_t = self.net_t(x_t)
        return self.classifier(torch.cat((feat_f, feat_t), dim=1))


# ==========================================
# 4. 학습 실행 함수 (날짜별 저장 포함)
# ==========================================
def main():
    seed_everything(CFG.SEED)

    # 선생님 로드
    teacher = TeacherNet().to(CFG.DEVICE)
    teacher.load_state_dict(torch.load(CFG.TEACHER_MODEL_PATH, map_location=CFG.DEVICE,weights_only=True))
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # 학생 준비
    student = StudentNet().to(CFG.DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)

    transform = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(QuadInputDataset(CFG.TRAIN_CSV, CFG.TRAIN_RGB_DIR, CFG.TRAIN_MASK_DIR, transform),
                              batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    dev_loader = DataLoader(QuadInputDataset(CFG.DEV_CSV, CFG.DEV_RGB_DIR, CFG.DEV_MASK_DIR, transform),
                            batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    best_dev_logloss = float('inf')
    patience_cnt = 0

    print(f"🚀 학생 모델(실전용 2채널) KD 학습 시작! (기기: {CFG.DEVICE})\n")

    print(f"\n🚀 [실시간 모니터링] 학생 모델 KD 학습 시작 (기기: {CFG.DEVICE})")
    print(f"{'Epoch':<5} | {'Tr_Loss':<8} | {'Dv_Loss':<8} | {'Tr_Acc':<7} | {'Dv_Acc':<7} | {'LogLoss':<8} | {'Gap'}")
    print("-" * 75)

    for epoch in range(CFG.EPOCHS):
        # [Train]
        student.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}", leave=False)

        for x_f, x_t, x_fm, x_tm, y in train_pbar:
            x_f, x_t, x_fm, x_tm, y = x_f.to(CFG.DEVICE), x_t.to(CFG.DEVICE), x_fm.to(CFG.DEVICE), x_tm.to(
                CFG.DEVICE), y.to(CFG.DEVICE)
            optimizer.zero_grad()
            with torch.autocast('cuda', enabled=CFG.USE_AMP):
                with torch.no_grad(): t_out = teacher(x_f, x_t, x_fm, x_tm)
                s_out = student(x_f, x_t)
                h_loss = F.cross_entropy(s_out, y)
                s_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(s_out / CFG.TEMPERATURE, dim=1),
                                                             F.softmax(t_out / CFG.TEMPERATURE, dim=1)) * (
                                     CFG.TEMPERATURE ** 2)
                loss = (CFG.ALPHA * h_loss) + ((1. - CFG.ALPHA) * s_loss)

            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update()
            train_loss += loss.item()
            train_correct += (s_out.argmax(1) == y).sum().item();
            train_total += y.size(0)

        # [Dev]
        student.eval()
        dev_loss, dev_preds, dev_trues, dev_correct, dev_total = 0, [], [], 0, 0
        with torch.no_grad():
            for x_f, x_t, _, _, y in dev_loader:
                x_f, x_t, y = x_f.to(CFG.DEVICE), x_t.to(CFG.DEVICE), y.to(CFG.DEVICE)
                out = student(x_f, x_t)
                dev_loss += F.cross_entropy(out, y).item()
                dev_preds.extend(torch.softmax(out, 1).cpu().numpy());
                dev_trues.extend(y.cpu().numpy())
                dev_correct += (out.argmax(1) == y).sum().item();
                dev_total += y.size(0)

        # 지표 계산
        tr_l, dv_l = train_loss / len(train_loader), dev_loss / len(dev_loader)
        tr_a, dv_a = train_correct / train_total, dev_correct / dev_total
        dev_logloss = LOGLOSS(np.eye(CFG.NUM_CLASSES)[np.array(dev_trues)], np.array(dev_preds))

        # ★ 과적합 감시 지표 (Train Acc와 Dev Acc의 차이)
        gap = tr_a - dv_a
        gap_str = f"⚠️ {gap:.2%}" if gap > 0.15 else f"OK({gap:.2%})"  # 차이가 15% 이상이면 경고 표시

        # 결과 한 줄 출력
        print(
            f"{epoch + 1:02>5} | {tr_l:<8.4f} | {dv_l:<8.4f} | {tr_a:<7.1%} | {dv_a:<7.1%} | {dev_logloss:<8.4f} | {gap_str}")

        if dev_logloss < best_dev_logloss:
            best_dev_logloss = dev_logloss;
            patience_cnt = 0
            t_date = datetime.now().strftime("%Y-%m-%d")
            save_path = os.path.join(CFG.SAVE_DIR, t_date)
            os.makedirs(save_path, exist_ok=True)
            torch.save(student.state_dict(), os.path.join(save_path, f"student_best_loss_{dev_logloss:.4f}.pth"))
        else:
            patience_cnt += 1
            if patience_cnt >= CFG.PATIENCE:
                print(f"\n📢 조기 종료: {CFG.PATIENCE}회 동안 LogLoss 개선 없음.")
                break

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()