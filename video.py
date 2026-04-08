import os
import cv2
import math
import numpy as np
import pandas as pd


DATA_ROOT = "./data"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VIDEO_NAME = "simulation.mp4"
OUTPUT_CSV = os.path.join(DATA_ROOT, "train_video_features.csv")


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
    # Otsu threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 배경/구조물 polarity 자동 보정
    # 구조물이 화면보다 작다고 가정하고 작은 쪽을 foreground로 사용
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

    # 0번은 background
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

    # 1) 시작-끝 차이
    diff_0_10 = frame_diff_score(gray_0, gray_10)

    # 2) 앞 3초 변화량 시계열
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

        # 영상별 상대 threshold
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

    # 3) mask 기반 구조물 형상 feature
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

    return {
        "diff_0_10": diff_0_10,
        "motion_0_3_mean": motion_0_3_mean,
        "motion_0_3_max": motion_0_3_max,
        "centroid_shift": centroid_shift,
        "height_drop_ratio": height_drop_ratio,
        "collapse_time_ratio": collapse_time_ratio,
    }


def main():
    rows = []

    sample_ids = sorted(os.listdir(TRAIN_DIR))
    for sample_id in sample_ids:
        sample_dir = os.path.join(TRAIN_DIR, sample_id)
        if not os.path.isdir(sample_dir):
            continue

        video_path = os.path.join(sample_dir, VIDEO_NAME)
        if not os.path.isfile(video_path):
            print(f"[skip] {sample_id}: no video")
            continue

        try:
            feats = extract_video_features(video_path)
            row = {"id": sample_id}
            row.update(feats)
            rows.append(row)
            print(f"[ok] {sample_id}")
        except Exception as e:
            print(f"[error] {sample_id}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nsaved: {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()