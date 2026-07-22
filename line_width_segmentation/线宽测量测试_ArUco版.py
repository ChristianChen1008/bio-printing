from pathlib import Path
import csv
import math

import cv2
import numpy as np

from line_width_measure import (
    allocate_samples,
    clean_mask,
    draw_annotation,
    measure_at,
    reject_outliers,
    sample_positions,
    summarize,
)


# 只需要改这两个值。
IMAGE_PATH = r"D:\bio-print\database\data\raw\images\test.jpg"

# 填你实际量到的 ArUco 黑色大方块外边长，不是白纸宽度。
MARKER_SIZE_MM = 19.2


# 一张图取多少个线宽点。
SAMPLES = 10
EDGE_MARGIN_FRACTION = 0.25

# 如果自动分割不好，可以改成 "dark" 或 "yellow_green" 试试。
LINE_MODE = "auto"

# 如果自动找打印线失败，可以手动填打印区域: x, y, 宽, 高。
# 不需要时保持 None。
LINE_BBOX = None

MIN_WIDTH_MM = None
MAX_WIDTH_MM = None

# 输出文件夹。可以改成你想保存结果的 D 盘文件夹。
# 例如: OUTPUT_DIR = Path(r"D:\bio-print\line_width_segmentation\measurement_outputs")
OUTPUT_DIR = Path(r"D:\bio-print\database\data\raw\images\annotated_pictures")

def aruco_dictionary():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "当前 Python 的 OpenCV 没有 aruco 模块。请安装 opencv-contrib-python。"
        )
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def detect_aruco_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = aruco_dictionary()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None or len(ids) == 0:
        raise RuntimeError(
            "没有识别到 ArUco 标记。请确认四个角完整、清楚、没有反光遮挡。"
        )

    ids_flat = ids.flatten()
    chosen_index = int(np.where(ids_flat == 0)[0][0]) if 0 in ids_flat else 0
    pts = corners[chosen_index].reshape(4, 2).astype(np.float32)

    edges = [
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[1] - pts[2]),
        np.linalg.norm(pts[2] - pts[3]),
        np.linalg.norm(pts[3] - pts[0]),
    ]
    side_px = float(np.mean(edges))
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    return {
        "id": int(ids_flat[chosen_index]),
        "corners": pts,
        "bbox": (int(x), int(y), int(w), int(h)),
        "side_px": side_px,
    }


def expand_bbox(bbox, pad, width, height):
    x, y, w, h = bbox
    x0 = max(0, int(x - pad))
    y0 = max(0, int(y - pad))
    x1 = min(width, int(x + w + pad))
    y1 = min(height, int(y + h + pad))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def component_candidates(mask, marker_bbox, marker_side_px, manual_bbox=None):
    height, width = mask.shape[:2]
    if manual_bbox is not None:
        x, y, w, h = manual_bbox
        region = np.zeros_like(mask, dtype=bool)
        region[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)] = True
        mask = mask & region

    marker_forbidden = expand_bbox(marker_bbox, int(marker_side_px * 0.35), width, height)
    marker_cx = marker_bbox[0] + marker_bbox[2] / 2.0
    marker_cy = marker_bbox[1] + marker_bbox[3] / 2.0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)

    candidates = []
    image_area = height * width
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        if area > image_area * 0.25:
            continue
        if w < marker_side_px * 0.35 or h < marker_side_px * 0.35:
            continue

        bbox = (x, y, w, h)
        if bbox_iou(bbox, marker_forbidden) > 0.02:
            continue

        bbox_area = max(1, w * h)
        fill = area / bbox_area
        aspect = max(w / max(1, h), h / max(1, w))
        if fill > 0.75 or aspect > 6:
            continue

        # 矩形打印轨迹通常是“大外框、低填充率”的连通区域。
        size_score = math.sqrt(bbox_area)
        fill_score = 1.0 / (1.0 + abs(fill - 0.12) * 5.0)
        aspect_score = 1.0 / math.sqrt(aspect)
        component_cx = x + w / 2.0
        component_cy = y + h / 2.0
        center_distance = math.hypot(component_cx - marker_cx, component_cy - marker_cy)
        proximity_score = 1.0 / (1.0 + (center_distance / max(1.0, marker_side_px * 2.0)) ** 2)
        score = size_score * fill_score * aspect_score * proximity_score
        candidates.append((score, label, bbox, area, fill, aspect))

    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates, labels


def make_line_masks(image, marker_bbox, marker_side_px, manual_bbox=None):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    marker_excluded = np.ones((height, width), dtype=bool)
    x, y, w, h = expand_bbox(marker_bbox, int(marker_side_px * 0.12), width, height)
    marker_excluded[y : y + h, x : x + w] = False

    masks = []

    yellow_green = (
        (hsv[:, :, 0] >= 18)
        & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 12)
        & (hsv[:, :, 2] >= 55)
    )
    masks.append(("yellow_green", yellow_green))

    color_excess = (
        (g.astype(np.int16) > b.astype(np.int16) + 4)
        & (r.astype(np.int16) > b.astype(np.int16) + 2)
        & (gray > 45)
    )
    masks.append(("color_excess", color_excess))

    masks.append(("dark", blur < otsu))
    masks.append(("bright", blur > otsu))

    cleaned = []
    for name, raw in masks:
        if LINE_MODE != "auto" and name != LINE_MODE:
            continue
        mask = clean_mask(raw & marker_excluded)
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
        cleaned.append((name, mask))
    return cleaned


def detect_line_mask(image, marker_bbox, marker_side_px):
    best = None
    best_labels = None
    best_name = None

    for name, mask in make_line_masks(image, marker_bbox, marker_side_px, LINE_BBOX):
        candidates, labels = component_candidates(mask, marker_bbox, marker_side_px, LINE_BBOX)
        if not candidates:
            continue
        candidate = candidates[0]
        if best is None or candidate[0] > best[0]:
            best = candidate
            best_labels = labels
            best_name = name

    if best is None:
        raise RuntimeError(
            "没有自动分割到打印线。请尝试调整 LINE_MODE，或临时填写 LINE_BBOX。"
        )

    _, label, bbox, _, _, _ = best
    component = best_labels == label
    return component, bbox, best_name


def contiguous_index_runs(indices):
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
        else:
            runs.append((start, previous))
            start = previous = value
    runs.append((start, previous))
    return runs


def estimate_frame_bbox(mask):
    ys, xs = np.where(mask)
    raw_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    row_counts = mask.sum(axis=1).astype(np.float32)
    col_counts = mask.sum(axis=0).astype(np.float32)
    row_smooth = np.convolve(row_counts, np.ones(11) / 11.0, mode="same")
    col_smooth = np.convolve(col_counts, np.ones(11) / 11.0, mode="same")

    row_threshold = max(8.0, float(row_smooth.max()) * 0.22)
    col_threshold = max(8.0, float(col_smooth.max()) * 0.18)
    row_runs = contiguous_index_runs(np.flatnonzero(row_smooth >= row_threshold))
    col_runs = contiguous_index_runs(np.flatnonzero(col_smooth >= col_threshold))

    row_runs = [run for run in row_runs if run[1] - run[0] + 1 >= 4]
    col_runs = [run for run in col_runs if run[1] - run[0] + 1 >= 4]
    if len(row_runs) < 2 or len(col_runs) < 2:
        return raw_bbox

    top_run = min(row_runs, key=lambda run: run[0])
    bottom_run = max(row_runs, key=lambda run: run[1])
    left_run = min(col_runs, key=lambda run: run[0])
    right_run = max(col_runs, key=lambda run: run[1])

    y0 = int(round((top_run[0] + top_run[1]) / 2.0))
    y1 = int(round((bottom_run[0] + bottom_run[1]) / 2.0))
    x0 = int(round((left_run[0] + left_run[1]) / 2.0))
    x1 = int(round((right_run[0] + right_run[1]) / 2.0))

    if x1 <= x0 or y1 <= y0:
        return raw_bbox
    return (x0, y0, x1, y1)


def build_edge_profile(mask, side, bbox):
    x0, y0, x1, y1 = bbox
    profile = []
    if side in ("top", "bottom"):
        for x in range(x0, x1 + 1):
            result = measure_at(mask, side, x, y0 if side == "top" else y1, bbox)
            if result:
                result["coord"] = x
                profile.append(result)
    else:
        for y in range(y0, y1 + 1):
            result = measure_at(mask, side, x0 if side == "left" else x1, y, bbox)
            if result:
                result["coord"] = y
                profile.append(result)
    return profile


def robust_width_limits(widths):
    widths = np.array(widths, dtype=np.float32)
    median = float(np.median(widths))
    mad = float(np.median(np.abs(widths - median)))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        lower = median * 0.55
        upper = median * 1.80
    else:
        lower = max(median * 0.45, median - 3.0 * sigma)
        upper = min(median * 2.20, median + 3.0 * sigma)
    return max(1.0, lower), max(2.0, upper)


def stable_segments_for_side(mask, side, bbox):
    profile = build_edge_profile(mask, side, bbox)
    if len(profile) < 8:
        return []

    widths = [row["width_px"] for row in profile]
    lower, upper = robust_width_limits(widths)
    valid_coords = [
        row["coord"]
        for row in profile
        if lower <= row["width_px"] <= upper
    ]
    coord_to_row = {row["coord"]: row for row in profile}
    runs = contiguous_index_runs(np.array(valid_coords, dtype=np.int32))

    x0, y0, x1, y1 = bbox
    span = (x1 - x0 + 1) if side in ("top", "bottom") else (y1 - y0 + 1)
    min_run_length = max(8, int(span * 0.08))
    segments = []
    for start, end in runs:
        selected = [
            coord_to_row[coord]
            for coord in range(start, end + 1)
            if coord in coord_to_row
        ]
        if len(selected) < min_run_length:
            continue
        selected_widths = np.array([row["width_px"] for row in selected], dtype=np.float32)
        mean_width = float(np.mean(selected_widths))
        std_width = float(np.std(selected_widths))
        cv = std_width / max(1e-6, mean_width)
        score = len(selected) / (1.0 + cv * 10.0)
        segments.append(
            {
                "side": side,
                "start": int(start),
                "end": int(end),
                "length": int(end - start + 1),
                "mean_width_px": mean_width,
                "std_width_px": std_width,
                "score": score,
            }
        )

    segments.sort(key=lambda item: item["score"], reverse=True)
    return segments


def choose_stable_segments(mask, bbox):
    segments = []
    for side in ("top", "right", "bottom", "left"):
        side_segments = stable_segments_for_side(mask, side, bbox)
        if side_segments:
            segments.append(side_segments[0])
    if not segments:
        return []
    segments.sort(key=lambda item: item["score"], reverse=True)
    return segments


def allocate_samples_to_segments(segments, total):
    if not segments:
        return []
    chosen = segments[:total]
    counts = np.ones(len(chosen), dtype=int)
    remaining = total - len(chosen)
    lengths = np.array([segment["length"] for segment in chosen], dtype=np.float32)
    weights = lengths / max(1.0, float(lengths.sum()))
    while remaining > 0:
        expected = weights * (total - len(chosen))
        extras = counts - 1
        idx = int(np.argmax(expected - extras))
        counts[idx] += 1
        remaining -= 1
    return list(zip(chosen, counts.tolist()))


def sample_stable_segments(mask, bbox):
    segments = choose_stable_segments(mask, bbox)
    allocations = allocate_samples_to_segments(segments, SAMPLES)
    samples = []
    for segment, count in allocations:
        side = segment["side"]
        start = segment["start"]
        end = segment["end"]
        margin = max(1, int((end - start + 1) * 0.15))
        positions = sample_positions(start, end, count, margin)
        for position in positions:
            position = int(position)
            if side in ("top", "bottom"):
                result = measure_at(mask, side, position, bbox[1] if side == "top" else bbox[3], bbox)
            else:
                result = measure_at(mask, side, bbox[0] if side == "left" else bbox[2], position, bbox)
            if result:
                result["stable_segment_start"] = start
                result["stable_segment_end"] = end
                samples.append(result)
    return samples


def measure_rectangle_widths(mask, image_path, mm_per_px):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("打印线 mask 为空。")

    x0, y0, x1, y1 = estimate_frame_bbox(mask)
    bbox = (x0, y0, x1, y1)

    samples = sample_stable_segments(mask, bbox)
    if not samples:
        horizontal_len = max(1, x1 - x0 + 1)
        vertical_len = max(1, y1 - y0 + 1)
        counts = allocate_samples(SAMPLES, horizontal_len, vertical_len)
        margin_x = max(2, int(horizontal_len * EDGE_MARGIN_FRACTION))
        margin_y = max(2, int(vertical_len * EDGE_MARGIN_FRACTION))

        for x in sample_positions(x0, x1, counts["top"], margin_x):
            result = measure_at(mask, "top", int(x), y0, bbox)
            if result:
                samples.append(result)
        for y in sample_positions(y0, y1, counts["right"], margin_y):
            result = measure_at(mask, "right", x1, int(y), bbox)
            if result:
                samples.append(result)
        for x in sample_positions(x0, x1, counts["bottom"], margin_x):
            result = measure_at(mask, "bottom", int(x), y1, bbox)
            if result:
                samples.append(result)
        for y in sample_positions(y0, y1, counts["left"], margin_y):
            result = measure_at(mask, "left", x0, int(y), bbox)
            if result:
                samples.append(result)

    rows = []
    for index, sample in enumerate(samples, start=1):
        rows.append(
            {
                "image": str(image_path),
                "sample_id": index,
                "side": sample["side"],
                "x": sample["x"],
                "y": sample["y"],
                "width_px": sample["width_px"],
                "width_mm": sample["width_px"] * mm_per_px,
                "kept": True,
                "reject_reason": "",
            }
        )
    return rows, bbox


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "sample_id",
            "side",
            "x",
            "y",
            "width_px",
            "width_mm",
            "kept",
            "reject_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_aruco_annotation(image, marker):
    annotated = image.copy()
    pts = marker["corners"].astype(np.int32)
    cv2.polylines(annotated, [pts], True, (255, 0, 0), 3)
    x, y, w, h = marker["bbox"]
    cv2.putText(
        annotated,
        f"ArUco id={marker['id']}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def main():
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError("请先修改 IMAGE_PATH 为真实图片路径。")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    marker = detect_aruco_marker(image)
    mm_per_px = MARKER_SIZE_MM / marker["side_px"]
    image_stem = image_path.stem
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"result_{image_stem}.csv"
    annotated_image = output_dir / f"annotated_{image_stem}.png"
    line_mask_image = output_dir / f"line_mask_{image_stem}.png"

    line_mask, line_component_bbox, chosen_line_mode = detect_line_mask(
        image, marker["bbox"], marker["side_px"]
    )
    rows, line_bbox = measure_rectangle_widths(line_mask, image_path, mm_per_px)
    rows = reject_outliers(rows, MIN_WIDTH_MM, MAX_WIDTH_MM)
    stats = summarize(rows)

    write_csv(rows, output_csv)
    cv2.imwrite(str(line_mask_image), (line_mask.astype(np.uint8) * 255))

    annotated = draw_annotation(image, rows, ref_bbox=marker["bbox"], line_bbox=line_bbox)
    annotated = draw_aruco_annotation(annotated, marker)
    cv2.imwrite(str(annotated_image), annotated)

    print("Done.")
    print(f"Image: {image_path}")
    print(f"ArUco id: {marker['id']}")
    print(f"Marker bbox: {marker['bbox']}")
    print(f"Marker side: {marker['side_px']:.2f} px")
    print(f"Marker size: {MARKER_SIZE_MM:.4f} mm")
    print(f"Scale: {mm_per_px:.8f} mm/pixel")
    print(f"Line mode: {chosen_line_mode}")
    print(f"Line bbox: {line_bbox}")
    print(f"Measured samples: {len(rows)}")
    print(f"Kept samples: {stats['n_kept']}")
    print(f"Mean width: {stats['mean_mm']:.6f} mm")
    print(f"Median width: {stats['median_mm']:.6f} mm")
    print(f"CSV: {output_csv}")
    print(f"Annotated image: {annotated_image}")
    print(f"Line mask: {line_mask_image}")


if __name__ == "__main__":
    main()
