import os
import cv2
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image(path, img):
    cv2.imwrite(path, img)


def normalize_to_uint8(img):
    img = np.abs(img)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img


def extract_edges(image_path, output_dir="./edge_results"):
    ensure_dir(output_dir)

    # 1. 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 원본 저장
    save_image(os.path.join(output_dir, f"{base_name}_original.png"), img)

    # 2. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. blur 버전들
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # -----------------------------
    # A. 기본 Canny
    # -----------------------------
    canny_basic = cv2.Canny(gaussian, 50, 150)
    save_image(os.path.join(output_dir, f"{base_name}_canny_basic.png"), canny_basic)

    # -----------------------------
    # B. 그림자/약한 외곽까지 좀 더 살리는 low-threshold Canny
    # -----------------------------
    canny_soft = cv2.Canny(gaussian, 20, 80)
    save_image(os.path.join(output_dir, f"{base_name}_canny_soft.png"), canny_soft)

    # -----------------------------
    # C. bilateral + Canny
    #    노이즈를 줄이면서 경계 보존
    # -----------------------------
    canny_bilateral = cv2.Canny(bilateral, 30, 100)
    save_image(os.path.join(output_dir, f"{base_name}_canny_bilateral.png"), canny_bilateral)

    # -----------------------------
    # D. Sobel magnitude
    #    부드러운 경계, 그림자 경계까지 볼 때 참고용
    # -----------------------------
    sobel_x = cv2.Sobel(gaussian, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gaussian, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = normalize_to_uint8(sobel_mag)
    save_image(os.path.join(output_dir, f"{base_name}_sobel_mag.png"), sobel_mag)

    # -----------------------------
    # E. Laplacian
    # -----------------------------
    lap = cv2.Laplacian(gaussian, cv2.CV_64F)
    lap = normalize_to_uint8(lap)
    save_image(os.path.join(output_dir, f"{base_name}_laplacian.png"), lap)

    # -----------------------------
    # F. Adaptive Threshold
    #    그림자/물체 전체 윤곽이 덩어리처럼 보이는지 확인용
    # -----------------------------
    adaptive = cv2.adaptiveThreshold(
        gaussian,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )
    save_image(os.path.join(output_dir, f"{base_name}_adaptive_thresh.png"), adaptive)

    # -----------------------------
    # G. Otsu Threshold
    #    전체 silhouette 확인용
    # -----------------------------
    _, otsu = cv2.threshold(
        gaussian, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    save_image(os.path.join(output_dir, f"{base_name}_otsu_thresh.png"), otsu)

    # -----------------------------
    # H. Morphological Gradient
    #    외곽 윤곽 강조
    # -----------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(gaussian, cv2.MORPH_GRADIENT, kernel)
    save_image(os.path.join(output_dir, f"{base_name}_morph_gradient.png"), morph_grad)

    print(f"[완료] 결과 저장 폴더: {output_dir}")


if __name__ == "__main__":
    image_path = "./object_whitebg.png"   # 여기에 테스트할 이미지 경로 넣기
    extract_edges(image_path, output_dir="./edge_results")
# 앳지

### 배경 없애기
# import cv2
# import numpy as np
# import os
#
#
# def extract_object_grabcut(
#     image_path: str,
#     output_mask_path: str = "object_mask.png",
#     output_object_path: str = "object_only.png",
#     output_whitebg_path: str = "object_whitebg.png",
#     rect_ratio=(0.30, 0.30, 0.40, 0.40),
#     iter_count: int = 10
# ):
#     """
#     GrabCut으로 중앙 물체를 분리하는 함수
#
#     rect_ratio:
#         (x_ratio, y_ratio, w_ratio, h_ratio)
#         이미지 크기 대비 GrabCut 초기 사각형 비율
#     """
#
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
#
#     h, w = img.shape[:2]
#
#     # GrabCut 초기 사각형 설정
#     rx = int(w * rect_ratio[0])
#     ry = int(h * rect_ratio[1])
#     rw = int(w * rect_ratio[2])
#     rh = int(h * rect_ratio[3])
#     rect = (rx, ry, rw, rh)
#
#     mask = np.zeros((h, w), np.uint8)
#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)
#
#     # GrabCut 실행
#     cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
#
#     # foreground만 1로
#     mask_bin = np.where(
#         (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
#         255,
#         0
#     ).astype("uint8")
#
#     # morphology로 작은 구멍/노이즈 정리
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
#     mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
#
#     # 가장 큰 connected component만 남기기
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
#     if num_labels > 1:
#         largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         largest_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
#     else:
#         largest_mask = mask_bin
#
#     # 물체만 남기기
#     object_only = cv2.bitwise_and(img, img, mask=largest_mask)
#
#     # 흰 배경 위에 물체만 올리기
#     white_bg = np.ones_like(img, dtype=np.uint8) * 255
#     white_bg[largest_mask == 255] = object_only[largest_mask == 255]
#
#     # 투명 배경 PNG 만들기 (RGBA)
#     b, g, r = cv2.split(img)
#     alpha = largest_mask
#     rgba = cv2.merge([b, g, r, alpha])
#
#     cv2.imwrite(output_mask_path, largest_mask)
#     cv2.imwrite(output_whitebg_path, white_bg)
#     cv2.imwrite(output_object_path, rgba)
#
#     print(f"[저장] mask: {output_mask_path}")
#     print(f"[저장] white background: {output_whitebg_path}")
#     print(f"[저장] transparent object png: {output_object_path}")
#
#
# if __name__ == "__main__":
#     image_path = "./data/train/TRAIN_0001/front.png"  # 입력 이미지
#     extract_object_grabcut(
#         image_path=image_path,
#         output_mask_path="object_mask.png",
#         output_object_path="object_only.png",
#         output_whitebg_path="object_whitebg.png",
#         rect_ratio=(0.30, 0.28, 0.40, 0.42),  # 중앙 물체에 맞게 조절 가능
#         iter_count=10
#     )