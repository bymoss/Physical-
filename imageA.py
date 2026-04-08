import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================
IMAGE_PATH = "./data/train/TRAIN_0420/front.png"   # 분석할 이미지 경로
SAVE_PATH = "opencv_preprocess_example.png"

# =========================
# 구조물 마스크 추출 함수
# =========================
def extract_structure_mask(img_bgr):
    """
    OpenCV 기반으로 구조물 영역을 대략 분리하는 예시 함수
    - HSV의 Saturation 채널 사용
    - Morphology로 노이즈 제거
    - 가장 큰 연결영역만 선택
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # saturation 기준 threshold
    # 필요하면 30~80 사이에서 조정
    _, mask = cv2.threshold(s, 45, 255, cv2.THRESH_BINARY)

    # morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 가장 큰 connected component만 남기기
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_idx] = 255

    return largest_mask

# =========================
# contour / bbox / centroid 시각화
# =========================
def draw_features(img_bgr, mask):
    vis = img_bgr.copy()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return vis, None

    cnt = max(contours, key=cv2.contourArea)

    # contour
    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

    # bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # centroid
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(vis, "centroid", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # bbox 중심도 같이 표시
    bx = x + w // 2
    by = y + h // 2
    cv2.circle(vis, (bx, by), 5, (255, 255, 0), -1)
    cv2.putText(vis, "bbox center", (bx + 8, by + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    feature_info = {
        "bbox": (x, y, w, h),
        "centroid": (cx, cy) if M["m00"] > 0 else None,
        "area": cv2.contourArea(cnt),
        "perimeter": cv2.arcLength(cnt, True),
    }

    return vis, feature_info

# =========================
# 실행
# =========================
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

mask = extract_structure_mask(img_bgr)
vis_img, feat = draw_features(img_bgr, mask)

# 마스크를 3채널로 바꿔서 보기 좋게
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

# BGR -> RGB 변환
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

# =========================
# PPT용 그림 저장
# =========================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image", fontsize=14, pad=12)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_rgb)
plt.title("Structure Mask", fontsize=14, pad=12)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(vis_rgb)
plt.title("Contour + BBox + Centroid", fontsize=14, pad=12)
plt.axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.93])   # 위쪽 7% 여백 확보
plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight", pad_inches=0.2)
plt.show()

if feat is not None:
    print("=== Extracted Feature Example ===")
    print("BBox (x, y, w, h):", feat["bbox"])
    print("Centroid (cx, cy):", feat["centroid"])
    print("Area:", feat["area"])
    print("Perimeter:", feat["perimeter"])

print(f"\n저장 완료: {SAVE_PATH}")