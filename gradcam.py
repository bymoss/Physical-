import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
from main import Net

# =========================
# 1) valid transform
# =========================
valid_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# =========================
# 2) 이미지 로드
# =========================
def load_rgb_image(img_path):
    img = Image.open(img_path).convert("RGB")
    return np.array(img)

def preprocess_image(img_rgb, transform):
    tensor = transform(image=img_rgb)["image"]
    return tensor.unsqueeze(0)   # [1, 3, H, W]


# =========================
# 3) 시각화용 reverse normalize
# =========================
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1)
    return img


# =========================
# 4) CAM overlay
# =========================
def overlay_cam_on_image(img_rgb, cam_map, alpha=0.4):
    """
    img_rgb: uint8 RGB image, shape [H, W, 3]
    cam_map: float [H, W], 0~1
    """
    heatmap = np.uint8(255 * cam_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.clip((1 - alpha) * img_rgb + alpha * heatmap, 0, 255).astype(np.uint8)
    return overlay


# =========================
# 5) 멀티뷰 모델용 Grad-CAM
# =========================
class MultiViewGradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def generate_cam(self, front_tensor, top_tensor, target_view="front", target_class=None):
        """
        target_view: 'front' or 'top'
        front_tensor, top_tensor: [1,3,H,W]
        """

        self.model.zero_grad()

        # target_view만 gradient 흐르게 하고, 다른 view는 detach
        if target_view == "front":
            inp_front = front_tensor.clone().detach().requires_grad_(True)
            inp_top = top_tensor.clone().detach()
        elif target_view == "top":
            inp_front = front_tensor.clone().detach()
            inp_top = top_tensor.clone().detach().requires_grad_(True)
        else:
            raise ValueError("target_view must be 'front' or 'top'")

        logits = self.model([inp_front, inp_top])   # [1,1]

        if target_class is None:
            score = logits[:, 0].sum()
        else:
            score = logits[:, target_class].sum()

        score.backward(retain_graph=True)

        # activations: [1, C, h, w]
        # gradients:   [1, C, h, w]
        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)        # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1, keepdim=True)       # [1, 1, h, w]
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(front_tensor.shape[2], front_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        prob = torch.sigmoid(logits)[0, 0].item()
        return cam, prob


# =========================
# 6) 한 샘플 실행
# =========================
def run_gradcam_for_sample(model, sample_dir, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)

    front_path = os.path.join(sample_dir, "front.png")
    top_path = os.path.join(sample_dir, "top.png")

    front_rgb = load_rgb_image(front_path)
    top_rgb = load_rgb_image(top_path)

    front_tensor = preprocess_image(front_rgb, valid_transform).to(device)
    top_tensor = preprocess_image(top_rgb, valid_transform).to(device)

    # target layer
    target_layer = model.backbone.blocks[-1]
    cam_extractor = MultiViewGradCAM(model, target_layer, device)

    model.eval()

    # front CAM
    cam_front, prob_front = cam_extractor.generate_cam(
        front_tensor, top_tensor, target_view="front"
    )

    # top CAM
    cam_top, prob_top = cam_extractor.generate_cam(
        front_tensor, top_tensor, target_view="top"
    )

    cam_extractor.remove_hooks()

    overlay_front = overlay_cam_on_image(front_rgb, cam_front)
    overlay_top = overlay_cam_on_image(top_rgb, cam_top)

    cv2.imwrite(os.path.join(save_dir, "front_cam.png"), cv2.cvtColor(overlay_front, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, "top_cam.png"), cv2.cvtColor(overlay_top, cv2.COLOR_RGB2BGR))

    # 4분할 보기
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(front_rgb)
    axes[0, 0].set_title("Front Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(overlay_front)
    axes[0, 1].set_title(f"Front Grad-CAM | prob={prob_front:.4f}")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(top_rgb)
    axes[1, 0].set_title("Top Original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_top)
    axes[1, 1].set_title(f"Top Grad-CAM | prob={prob_top:.4f}")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cam_grid.png"), dpi=200)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)

ckpt_path = "./model_5fold/best_model_fold5.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

sample_dir = "./data/train/TRAIN_0167"   # 네 샘플 경로에 맞게 수정
save_dir = "./gradcam_results/sample17"

run_gradcam_for_sample(model, sample_dir, save_dir, device)