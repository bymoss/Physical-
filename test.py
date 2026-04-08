import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.local_global_model import LocalGlobalNet


CFG = {
    'IMG_SIZE': 384,
    'BATCH_SIZE': 8,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# transform
# =========================
test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


# =========================
# test dataset
# =========================
class TestLocalGlobalDataset(Dataset):
    def __init__(
        self,
        df,
        global_root,
        local_root,
        transform=None
    ):
        self.df = df.reset_index(drop=True)
        self.global_root = global_root
        self.local_root = local_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_id = str(self.df.iloc[idx]['id'])

        global_folder = os.path.join(self.global_root, sample_id)
        local_folder = os.path.join(self.local_root, sample_id)

        front_global = np.array(Image.open(os.path.join(global_folder, "front.png")).convert("RGB"))
        top_global = np.array(Image.open(os.path.join(global_folder, "top.png")).convert("RGB"))
        front_local = np.array(Image.open(os.path.join(local_folder, "front.png")).convert("RGB"))
        top_local = np.array(Image.open(os.path.join(local_folder, "top.png")).convert("RGB"))

        if self.transform is not None:
            front_global = self.transform(image=front_global)['image']
            top_global = self.transform(image=top_global)['image']
            front_local = self.transform(image=front_local)['image']
            top_local = self.transform(image=top_local)['image']

        views = {
            'front_global': front_global,
            'top_global': top_global,
            'front_local': front_local,
            'top_local': top_local,
        }

        return views


def move_views_to_device(views, device):
    return {k: v.to(device) for k, v in views.items()}


# =========================
# inference
# =========================
test_df = pd.read_csv('./data/sample_submission.csv')

test_dataset = TestLocalGlobalDataset(
    df=test_df,
    global_root='./data/test',                 # 원본 test 이미지
    local_root='./data/test_center_masked',    # center-masked test 이미지
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG['BATCH_SIZE'],
    shuffle=False,
    num_workers=0
)

model = LocalGlobalNet(num_classes=1).to(device)

best_model_path = './student_kd_finetune_ckpt/best_student_kd_finetune3.pth'
checkpoint = torch.load(best_model_path, map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

predictions = []

with torch.no_grad():
    for views in test_loader:
        views = move_views_to_device(views, device)

        outputs = model(views)                 # [B, 1]
        probs = torch.sigmoid(outputs)         # unstable probability
        predictions.extend(probs.cpu().numpy().flatten())

submission = pd.read_csv('./data/sample_submission.csv')
submission['unstable_prob'] = predictions
submission['stable_prob'] = 1 - submission['unstable_prob']

submission.to_csv('./final_submission.csv', index=False, encoding='utf-8-sig')
print("🎉 평가 완료! [final_submission.csv] 파일이 생성되었습니다.")