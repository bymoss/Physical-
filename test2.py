import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime

# ==========================================
# 1. 설정 클래스 (CFG)
# ==========================================
class CFG:
    TEST_RGB_DIR: str = "./data/test"  # 테스트 이미지 루트 폴더
    # ★ [필수 수정] 학습된 학생 모델(.pth)의 경로를 입력하세요
    STUDENT_MODEL_PATH: str = "./model_student/2026-03-28/student_best_loss_0.5000.pth"

    IMG_SIZE: int = 224
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    IMAGE_EXT: str = "png"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES: int = 2


# ==========================================
# 2. 테스트 데이터셋 클래스
# ==========================================
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 테스트 폴더 내의 모든 하위 폴더 목록 (예: test_00001, test_00002...)
        self.folder_list = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        folder_name = self.folder_list[idx]
        folder_path = os.path.join(self.root_dir, folder_name)

        # 정면과 윗면 사진 로드 (실전 규칙: 원본만 사용)
        img_f = Image.open(os.path.join(folder_path, f"front.{CFG.IMAGE_EXT}")).convert('RGB')
        img_t = Image.open(os.path.join(folder_path, f"top.{CFG.IMAGE_EXT}")).convert('RGB')

        if self.transform:
            img_f = self.transform(img_f)
            img_t = self.transform(img_t)

        return img_f, img_t, folder_name


# ==========================================
# 3. 학생 모델 구조 (학습 때와 동일해야 함)
# ==========================================
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.net_f = models.efficientnet_b0(weights=None)
        self.net_t = models.efficientnet_b0(weights=None)
        in_features = self.net_f.classifier[1].in_features
        self.net_f.classifier = nn.Identity()
        self.net_t.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features * 2, CFG.NUM_CLASSES)

    def forward(self, x_f, x_t):
        feat_f = self.net_f(x_f)
        feat_t = self.net_t(x_t)
        return self.classifier(torch.cat((feat_f, feat_t), dim=1))


# ==========================================
# 4. 예측 실행 (Inference)
# ==========================================
# ... (상단 설정 및 모델 클래스는 이전과 동일) ...

def run_inference():
    print(f"🔍 실전 데이터 예측 시작 (양식: id, unstable_prob, stable_prob)")

    # 1. 모델 로드 (가중치 로드 시 에러 방지용 weights_only=True)
    model = StudentNet().to(CFG.DEVICE)
    model.load_state_dict(torch.load(CFG.STUDENT_MODEL_PATH, map_location=CFG.DEVICE, weights_only=True))
    model.eval()

    # 2. 데이터 로더
    transform = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = TestDataset(CFG.TEST_RGB_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    results = []
    with torch.no_grad():
        for img_f, img_t, folder_names in tqdm(test_loader, desc="Predicting"):
            img_f, img_t = img_f.to(CFG.DEVICE), img_t.to(CFG.DEVICE)

            # 예측값 계산
            outputs = model(img_f, img_t)

            # 확신도를 높이고 싶다면 T(Temperature) 적용 (0.5~1.0 사이 조절)
            T = 0.7
            probs = torch.softmax(outputs / T, dim=1).cpu().numpy()

            # 순서 주의: 보통 0번 인덱스가 Stable, 1번이 Unstable입니다.
            # 만약 학습 시 반대로 하셨다면 [:, 1]과 [:, 0]을 바꾸세요.
            for i, name in enumerate(folder_names):
                s_prob = probs[i, 0]  # Stable 확률
                u_prob = probs[i, 1]  # Unstable 확률
                results.append([name, u_prob, s_prob])

    # 3. 요구하는 컬럼명으로 CSV 저장
    submission = pd.DataFrame(results, columns=['id', 'unstable_prob', 'stable_prob'])

    save_name = f"final_submission_{datetime.now().strftime('%H%M%S')}.csv"
    submission.to_csv(save_name, index=False)
    print(f"\n✅ 파일 저장 완료: {save_name}")


if __name__ == '__main__':
    run_inference()