import os
import cv2
import numpy as np
from pathlib import Path


# =========================
# 기본 유틸
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_image_file(filename: str) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return Path(filename).suffix.lower() in exts


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.abs(img)
    img = img - img.min()
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    img = (img * 255).astype(np.uint8)
    return img


# =========================
# 1. 물체 분리 (GrabCut)
# =========================
def extract_object_grabcut(
    image_bgr: np.ndarray,
    rect_ratio=(0.30, 0.28, 0.40, 0.42),
    iter_count: int = 10,
    bg_color=(220, 220, 220),
):
    """
    입력:
        image_bgr: 원본 BGR 이미지
    출력:
        object_whitebg: 배경을 단색으로 바꾼 컬러 이미지
        largest_mask: 물체 마스크 (uint8, 0 or 255)
        rgba_object: 투명 배경 포함 RGBA 이미지
    """
    h, w = image_bgr.shape[:2]

    rx = int(w * rect_ratio[0])
    ry = int(h * rect_ratio[1])
    rw = int(w * rect_ratio[2])
    rh = int(h * rect_ratio[3])
    rect = (rx, ry, rw, rh)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image_bgr,
        mask,
        rect,
        bgd_model,
        fgd_model,
        iter_count,
        cv2.GC_INIT_WITH_RECT
    )

    mask_bin = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # morphology로 노이즈 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

    # 가장 큰 connected component만 남기기
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
    else:
        largest_mask = mask_bin

    object_only = cv2.bitwise_and(image_bgr, image_bgr, mask=largest_mask)

    object_whitebg = np.full_like(image_bgr, bg_color, dtype=np.uint8)
    object_whitebg[largest_mask == 255] = object_only[largest_mask == 255]

    b, g, r = cv2.split(image_bgr)
    alpha = largest_mask
    rgba_object = cv2.merge([b, g, r, alpha])

    return object_whitebg, largest_mask, rgba_object


# =========================
# 2. 원본 이미지에서 Sobel Edge 생성
# =========================
def make_sobel_mag(
    image_bgr: np.ndarray,
    blur: bool = True,
    blur_ksize: int = 5,
    sobel_ksize: int = 3
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if blur:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = normalize_to_uint8(sobel_mag)
    return sobel_mag


# =========================
# 3. 단일 이미지 처리
# =========================
def process_one_image(
    image_path: str,
    out_object_whitebg_path: str,
    out_edge_original_sobel_path: str,
    out_mask_path: str = None,
    out_rgba_path: str = None,
    rect_ratio=(0.30, 0.28, 0.40, 0.42),
    iter_count: int = 10,
    bg_color=(220, 220, 220),
    blur: bool = True,
    blur_ksize: int = 5,
    sobel_ksize: int = 3,
):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[건너뜀] 읽기 실패: {image_path}")
        return

    # 1) 배경 제거 컬러 이미지
    object_whitebg, mask_img, rgba_object = extract_object_grabcut(
        img,
        rect_ratio=rect_ratio,
        iter_count=iter_count,
        bg_color=bg_color,
    )

    # 2) 원본 이미지 기준 Sobel
    sobel_edge_original = make_sobel_mag(
        img,
        blur=blur,
        blur_ksize=blur_ksize,
        sobel_ksize=sobel_ksize,
    )

    ensure_dir(os.path.dirname(out_object_whitebg_path))
    ensure_dir(os.path.dirname(out_edge_original_sobel_path))

    cv2.imwrite(out_object_whitebg_path, object_whitebg)
    cv2.imwrite(out_edge_original_sobel_path, sobel_edge_original)

    if out_mask_path is not None:
        ensure_dir(os.path.dirname(out_mask_path))
        cv2.imwrite(out_mask_path, mask_img)

    if out_rgba_path is not None:
        ensure_dir(os.path.dirname(out_rgba_path))
        cv2.imwrite(out_rgba_path, rgba_object)

    print(f"[완료] {image_path}")


# =========================
# 4. 폴더 전체 처리
# =========================
def process_dataset(
    input_root="./data",
    output_root="./processed",
    target_image_names=None,
    rect_ratio=(0.30, 0.28, 0.40, 0.42),
    iter_count=10,
    bg_color=(220, 220, 220),
    blur=True,
    blur_ksize=5,
    sobel_ksize=3,
    save_mask=False,
    save_rgba=False,
):
    """
    예:
        ./data/train/TRAIN001/front.png
        ./data/train/TRAIN001/top.png
        ./data/val/VAL001/front.png
        ./data/test/TEST001/top.png
    """
    if target_image_names is None:
        target_image_names = {"front.png", "top.png"}

    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)

    object_root = os.path.join(output_root, "object_whitebg")
    edge_original_root = os.path.join(output_root, "edge_sobel_original")
    mask_root = os.path.join(output_root, "object_mask")
    rgba_root = os.path.join(output_root, "object_rgba")

    total_count = 0

    for root, _, files in os.walk(input_root):
        for fname in files:
            if not is_image_file(fname):
                continue

            if fname not in target_image_names:
                continue

            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, input_root)

            out_object_whitebg_path = os.path.join(object_root, rel_path)
            out_edge_original_sobel_path = os.path.join(edge_original_root, rel_path)

            out_mask_path = None
            if save_mask:
                out_mask_path = os.path.join(mask_root, rel_path)

            out_rgba_path = None
            if save_rgba:
                rel_png = str(Path(rel_path).with_suffix(".png"))
                out_rgba_path = os.path.join(rgba_root, rel_png)

            process_one_image(
                image_path=in_path,
                out_object_whitebg_path=out_object_whitebg_path,
                out_edge_original_sobel_path=out_edge_original_sobel_path,
                out_mask_path=out_mask_path,
                out_rgba_path=out_rgba_path,
                rect_ratio=rect_ratio,
                iter_count=iter_count,
                bg_color=bg_color,
                blur=blur,
                blur_ksize=blur_ksize,
                sobel_ksize=sobel_ksize,
            )
            total_count += 1

    print("=" * 60)
    print(f"[전체 완료] 총 처리 이미지 수: {total_count}")
    print(f"[배경 제거 컬러] {object_root}")
    print(f"[원본 Sobel]      {edge_original_root}")
    if save_mask:
        print(f"[마스크]          {mask_root}")
    if save_rgba:
        print(f"[RGBA]            {rgba_root}")
    print("=" * 60)


# =========================
# 5. 실행부
# =========================
if __name__ == "__main__":
    process_dataset(
        input_root="./data/test",
        output_root="./processed",

        # 처리할 파일명
        target_image_names={"front.png", "top.png"},

        # GrabCut 설정
        rect_ratio=(0.30, 0.28, 0.40, 0.42),
        iter_count=10,

        # 배경 제거 후 배경색
        bg_color=(220, 220, 220),

        # 원본 Sobel 설정
        blur=True,
        blur_ksize=5,
        sobel_ksize=3,

        # 필요하면 True로
        save_mask=False,
        save_rgba=False,
    )