from pathlib import Path
import csv
import math

import cv2
import numpy as np


# ===================== 你主要改这里 =====================

# 输入图片路径。可以改成你的任意一张图片。
IMAGE_PATH = r"D:\bio-print\database\data\raw\images\test.jpg"

# ArUco 黑色大方块的真实边长，单位 mm。
# 注意：填黑色方块外边缘到外边缘的尺寸，不是白色纸片总宽度。
MARKER_SIZE_MM = 16

# Marker dictionary.
# Your current purchased/used marker in 25.jpg is APRILTAG_36H11 ID 0.
# If you later use the old generated ArUco marker, change this back to "DICT_4X4_50".
MARKER_DICTIONARY_NAME = "DICT_APRILTAG_36H11"

# 输出文件夹。程序会生成：
# annotated_原图名.png、line_mask_原图名.png、result_原图名.csv
OUTPUT_DIR = Path(r"D:\bio-print\database\data\raw\images\annotated_pictures")

# 每张图一共取多少个宽度点。
SAMPLES = 10

# Use only the single most stable continuous run among the selected segments.
SINGLE_BEST_STABLE_SEGMENT = True
MIN_VALID_SAMPLES = 7
MAX_WIDTH_RANGE_RATIO = 0.50

# 选择要测量的段编号：
# 1=右边下半段，2=右边上半段，3=上边右半段，4=上边左半段
# 5=左边上半段，6=左边下半段，7=下边左半段，8=下边右半段
# 可以写 "1"，也可以写 "1,2,8"。
# 留空 "" 时，程序运行后会在 VS Code 终端里让你输入。
SELECTED_SEGMENTS = ""

# 如果自动分割不理想，可改成 "yellow_green"、"dark"、"bright" 或 "color_excess"。
LINE_MODE = "auto"

# 如果自动找打印图案区域失败，可以临时手动填 x, y, 宽, 高；正常保持 None。
LINE_BBOX = None

# 可选：按经验剔除极端线宽。单位 mm；不需要就保持 None。
MIN_WIDTH_MM = None
MAX_WIDTH_MM = None

# Coordinate mode:
# "click_two_corners": click the real (0, 0) origin, then click the top-left corner.
# "manual_two_corners": use MANUAL_ORIGIN_PX and MANUAL_TOP_LEFT_PX directly.
# "click_origin_30mm": click the real (0, 0) origin, then build a 30 mm square.
# "manual_origin_30mm": use MANUAL_ORIGIN_PX directly, no click window.
# "auto_bbox": old method, estimate the rectangle from the printed line mask.
COORDINATE_MODE = "click_two_corners"
PATTERN_SIZE_MM = 30.0
MANUAL_ORIGIN_PX = None
MANUAL_TOP_LEFT_PX = None
MANUAL_X_AXIS_PX = None
MANUAL_Y_AXIS_PX = None

# When measuring one edge, only search near that edge.
# This prevents the left edge from accidentally grabbing the right edge/tail.
EDGE_SEARCH_BAND_MM = 6.0


# ===================== 基础工具函数 =====================

def clean_mask(mask):
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    return mask.astype(bool)


def contiguous_runs(indices):
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


def sample_positions(start, end, count, margin):
    usable_start = int(start + margin)
    usable_end = int(end - margin)
    if usable_end <= usable_start:
        usable_start, usable_end = int(start), int(end)
    return np.linspace(usable_start, usable_end, count + 2, dtype=int)[1:-1]


def measure_at(mask, side, x, y, bbox, search_radius_px=None):
    x0, y0, x1, y1 = bbox
    if side in ("top", "bottom"):
        if x < x0 or x > x1:
            return None
        if search_radius_px is None:
            search_y0, search_y1 = y0, y1
        elif side == "top":
            search_y0 = y0
            search_y1 = min(y1, y0 + int(search_radius_px))
        else:
            search_y0 = max(y0, y1 - int(search_radius_px))
            search_y1 = y1
        ys = np.flatnonzero(mask[search_y0 : search_y1 + 1, x]) + search_y0
        runs = contiguous_runs(ys)
        if not runs:
            return None
        run = min(runs, key=lambda r: r[0]) if side == "top" else max(runs, key=lambda r: r[1])
        center = (run[0] + run[1]) / 2.0
        return {
            "side": side,
            "x": int(x),
            "y": int(round(center)),
            "width_px": float(run[1] - run[0] + 1),
        }

    if y < y0 or y > y1:
        return None
    if search_radius_px is None:
        search_x0, search_x1 = x0, x1
    elif side == "left":
        search_x0 = x0
        search_x1 = min(x1, x0 + int(search_radius_px))
    else:
        search_x0 = max(x0, x1 - int(search_radius_px))
        search_x1 = x1
    xs = np.flatnonzero(mask[y, search_x0 : search_x1 + 1]) + search_x0
    runs = contiguous_runs(xs)
    if not runs:
        return None
    run = min(runs, key=lambda r: r[0]) if side == "left" else max(runs, key=lambda r: r[1])
    center = (run[0] + run[1]) / 2.0
    return {
        "side": side,
        "x": int(round(center)),
        "y": int(y),
        "width_px": float(run[1] - run[0] + 1),
    }


def reject_outliers(rows, min_mm=None, max_mm=None):
    for row in rows:
        value = row["width_mm"]
        if min_mm is not None and value < min_mm:
            row["kept"] = False
            row["reject_reason"] = "below_min"
        elif max_mm is not None and value > max_mm:
            row["kept"] = False
            row["reject_reason"] = "above_max"
        else:
            row["kept"] = True
            row["reject_reason"] = ""

    candidate_values = np.array([r["width_mm"] for r in rows if r["kept"]], dtype=float)
    if len(candidate_values) >= 5:
        median = float(np.median(candidate_values))
        mad = float(np.median(np.abs(candidate_values - median)))
        if mad > 1e-9:
            for row in rows:
                if not row["kept"]:
                    continue
                robust_z = 0.6745 * abs(row["width_mm"] - median) / mad
                if robust_z > 3.5:
                    row["kept"] = False
                    row["reject_reason"] = "outlier"
    return rows


def summarize(rows):
    kept_values = np.array([r["width_mm"] for r in rows if r["kept"]], dtype=float)
    if len(kept_values) == 0:
        return {"n_kept": 0, "mean_mm": math.nan, "std_mm": math.nan, "median_mm": math.nan}
    return {
        "n_kept": int(len(kept_values)),
        "mean_mm": float(np.mean(kept_values)),
        "std_mm": float(np.std(kept_values, ddof=1)) if len(kept_values) > 1 else 0.0,
        "median_mm": float(np.median(kept_values)),
    }


def parse_selected_segments(text):
    text = str(text).strip()
    if not text:
        print()
        print("请输入要测量的段编号，例如：1 或 1,2,8")
        print("1=右下  2=右上  3=上右  4=上左  5=左上  6=左下  7=下左  8=下右")
        text = input("段编号: ").strip()

    text = text.replace("，", ",").replace(" ", ",")
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1 or value > 8:
            raise ValueError("段编号只能是 1 到 8。")
        if value not in values:
            values.append(value)

    if not values:
        raise ValueError("至少需要选择一个段编号。")
    return values


# ===================== ArUco 定标 =====================

def aruco_dictionary():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("当前 OpenCV 没有 aruco 模块，请安装 opencv-contrib-python。")
    if not hasattr(cv2.aruco, MARKER_DICTIONARY_NAME):
        raise RuntimeError(f"当前 OpenCV 不支持这个标记字典: {MARKER_DICTIONARY_NAME}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICTIONARY_NAME))


def detect_aruco_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = aruco_dictionary()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None or len(ids) == 0:
        raise RuntimeError("没有识别到 ArUco 标记，请确认四个角完整、清晰、没有强反光遮挡。")

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


# ===================== 打印线自动分割 =====================

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
        if fill > 0.75 or aspect > 7:
            continue

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

    yellow_green = (
        (hsv[:, :, 0] >= 18)
        & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 12)
        & (hsv[:, :, 2] >= 55)
    )
    color_excess = (
        (g.astype(np.int16) > b.astype(np.int16) + 4)
        & (r.astype(np.int16) > b.astype(np.int16) + 2)
        & (gray > 45)
    )

    raw_masks = [
        ("yellow_green", yellow_green),
        ("color_excess", color_excess),
        ("dark", blur < otsu),
        ("bright", blur > otsu),
    ]

    cleaned = []
    for name, raw in raw_masks:
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
        raise RuntimeError("没有自动分割到打印线，请尝试调整 LINE_MODE，或临时填写 LINE_BBOX。")

    _, label, bbox, _, _, _ = best
    component = best_labels == label
    return component, bbox, best_name


# ===================== 八段坐标系测量逻辑 =====================

def estimate_frame_bbox(mask):
    ys, xs = np.where(mask)
    raw_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    row_counts = mask.sum(axis=1).astype(np.float32)
    col_counts = mask.sum(axis=0).astype(np.float32)
    row_smooth = np.convolve(row_counts, np.ones(11) / 11.0, mode="same")
    col_smooth = np.convolve(col_counts, np.ones(11) / 11.0, mode="same")

    row_threshold = max(8.0, float(row_smooth.max()) * 0.22)
    col_threshold = max(8.0, float(col_smooth.max()) * 0.18)
    row_runs = contiguous_runs(np.flatnonzero(row_smooth >= row_threshold))
    col_runs = contiguous_runs(np.flatnonzero(col_smooth >= col_threshold))

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


def build_segment_defs(bbox):
    x0, y0, x1, y1 = bbox
    xm = int(round((x0 + x1) / 2.0))
    ym = int(round((y0 + y1) / 2.0))
    return {
        1: {"side": "right", "start": ym, "end": y1, "name": "right_lower", "p1": (x1, y1), "p2": (x1, ym), "coord_start": (0, 0), "coord_end": (0, 15)},
        2: {"side": "right", "start": y0, "end": ym, "name": "right_upper", "p1": (x1, ym), "p2": (x1, y0), "coord_start": (0, 15), "coord_end": (0, 30)},
        3: {"side": "top", "start": xm, "end": x1, "name": "top_right", "p1": (x1, y0), "p2": (xm, y0), "coord_start": (0, 30), "coord_end": (15, 30)},
        4: {"side": "top", "start": x0, "end": xm, "name": "top_left", "p1": (xm, y0), "p2": (x0, y0), "coord_start": (15, 30), "coord_end": (30, 30)},
        5: {"side": "left", "start": y0, "end": ym, "name": "left_upper", "p1": (x0, y0), "p2": (x0, ym), "coord_start": (30, 30), "coord_end": (30, 15)},
        6: {"side": "left", "start": ym, "end": y1, "name": "left_lower", "p1": (x0, ym), "p2": (x0, y1), "coord_start": (30, 15), "coord_end": (30, 0)},
        7: {"side": "bottom", "start": x0, "end": xm, "name": "bottom_left", "p1": (x0, y1), "p2": (xm, y1), "coord_start": (30, 0), "coord_end": (15, 0)},
        8: {"side": "bottom", "start": xm, "end": x1, "name": "bottom_right", "p1": (xm, y1), "p2": (x1, y1), "coord_start": (15, 0), "coord_end": (0, 0)},
    }


def build_segment_profile(mask, segment, bbox, search_radius_px):
    side = segment["side"]
    profile = []
    for coord in range(int(segment["start"]), int(segment["end"]) + 1):
        if side in ("top", "bottom"):
            result = measure_at(
                mask,
                side,
                coord,
                bbox[1] if side == "top" else bbox[3],
                bbox,
                search_radius_px,
            )
        else:
            result = measure_at(
                mask,
                side,
                bbox[0] if side == "left" else bbox[2],
                coord,
                bbox,
                search_radius_px,
            )
        if result:
            result["coord"] = coord
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


def stable_segments_inside_selected(mask, bbox, selected_numbers, search_radius_px):
    segment_defs = build_segment_defs(bbox)
    stable_segments = []

    for number in selected_numbers:
        segment = segment_defs[number].copy()
        profile = build_segment_profile(mask, segment, bbox, search_radius_px)
        if len(profile) < 5:
            continue

        widths = [row["width_px"] for row in profile]
        lower, upper = robust_width_limits(widths)
        coord_to_row = {row["coord"]: row for row in profile}
        valid_coords = [
            row["coord"]
            for row in profile
            if lower <= row["width_px"] <= upper
        ]

        runs = contiguous_runs(np.array(valid_coords, dtype=np.int32))
        span = int(segment["end"] - segment["start"] + 1)
        min_run_length = max(5, int(span * 0.12))
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
            stable = segment.copy()
            stable.update(
                {
                    "number": number,
                    "stable_start": int(start),
                    "stable_end": int(end),
                    "length": int(end - start + 1),
                    "mean_width_px": mean_width,
                    "std_width_px": std_width,
                    "score": score,
                }
            )
            stable_segments.append(stable)

    stable_segments.sort(key=lambda item: item["score"], reverse=True)
    return stable_segments


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


def fallback_segments(selected_numbers, bbox):
    result = []
    for number in selected_numbers:
        segment = build_segment_defs(bbox)[number].copy()
        segment.update(
            {
                "number": number,
                "stable_start": int(segment["start"]),
                "stable_end": int(segment["end"]),
                "length": int(segment["end"] - segment["start"] + 1),
                "mean_width_px": "",
                "std_width_px": "",
                "score": 0.0,
            }
        )
        result.append(segment)
    return result


def sample_selected_segments(mask, bbox, selected_numbers, mm_per_px):
    search_radius_px = max(8, int(round(EDGE_SEARCH_BAND_MM / mm_per_px)))
    stable_segments = stable_segments_inside_selected(mask, bbox, selected_numbers, search_radius_px)
    used_fallback = False
    if not stable_segments:
        stable_segments = fallback_segments(selected_numbers, bbox)
        used_fallback = True

    if SINGLE_BEST_STABLE_SEGMENT:
        stable_segments = stable_segments[:1]

    allocations = allocate_samples_to_segments(stable_segments, SAMPLES)
    used_segments = [segment for segment, _ in allocations]

    samples = []
    for segment, count in allocations:
        side = segment["side"]
        start = int(segment["stable_start"])
        end = int(segment["stable_end"])
        margin = max(1, int((end - start + 1) * 0.15))
        for position in sample_positions(start, end, count, margin):
            position = int(position)
            if side in ("top", "bottom"):
                result = measure_at(
                    mask,
                    side,
                    position,
                    bbox[1] if side == "top" else bbox[3],
                    bbox,
                    search_radius_px,
                )
            else:
                result = measure_at(
                    mask,
                    side,
                    bbox[0] if side == "left" else bbox[2],
                    position,
                    bbox,
                    search_radius_px,
                )
            if result:
                result["segment_number"] = segment["number"]
                result["segment_name"] = segment["name"]
                result["segment_coord_start"] = str(segment["coord_start"])
                result["segment_coord_end"] = str(segment["coord_end"])
                result["stable_segment_start"] = start
                result["stable_segment_end"] = end
                samples.append(result)
    return samples, used_segments, used_fallback


def select_origin_by_click(image):
    max_display_width = 1200
    max_display_height = 900
    height, width = image.shape[:2]
    scale = min(max_display_width / width, max_display_height / height, 1.0)
    display_size = (int(width * scale), int(height * scale))
    display_base = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
    state = {"point": None}
    window_name = "click origin, then press Enter"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (int(round(x / scale)), int(round(y / scale)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print()
    print("请在弹出的图片窗口中，用鼠标左键点击你定义的 (0,0) 原点。")
    print("点好以后按 Enter 确认；如果点错了，可以重新点。按 Esc 取消。")

    while True:
        display = display_base.copy()
        cv2.putText(
            display,
            "Click origin, press Enter",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if state["point"] is not None:
            px = int(round(state["point"][0] * scale))
            py = int(round(state["point"][1] * scale))
            cv2.circle(display, (px, py), 8, (255, 0, 255), -1)
            cv2.putText(
                display,
                f"origin=({state['point'][0]}, {state['point'][1]})",
                (px + 10, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if state["point"] is None:
                print("还没有点击原点，请先在图片上点一下。")
                continue
            cv2.destroyWindow(window_name)
            return state["point"]
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("你取消了原点选择。")


def select_two_corners_by_click(image):
    max_display_width = 1200
    max_display_height = 900
    height, width = image.shape[:2]
    scale = min(max_display_width / width, max_display_height / height, 1.0)
    display_size = (int(width * scale), int(height * scale))
    display_base = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
    state = {"points": []}
    window_name = "click origin and top-left, then press Enter"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (int(round(x / scale)), int(round(y / scale)))
            if len(state["points"]) >= 2:
                state["points"] = []
            state["points"].append(point)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print()
    print("请在弹出的图片窗口中依次点击两个点：")
    print("第 1 下：右下角 (0,0) 原点；第 2 下：左上角 (30,30) 对应点。")
    print("两个点都点好后按 Enter 确认；如果点错，继续点击会重新开始。按 Esc 取消。")

    while True:
        display = display_base.copy()
        cv2.putText(
            display,
            "1) origin  2) top-left  Enter=OK",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        display_points = []
        for point in state["points"]:
            display_points.append((int(round(point[0] * scale)), int(round(point[1] * scale))))

        if len(display_points) >= 1:
            cv2.circle(display, display_points[0], 8, (255, 0, 255), -1)
            cv2.putText(
                display,
                "origin",
                (display_points[0][0] + 10, display_points[0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if len(display_points) >= 2:
            cv2.circle(display, display_points[1], 8, (255, 255, 0), -1)
            cv2.putText(
                display,
                "top-left",
                (display_points[1][0] + 10, display_points[1][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(display, display_points[1], display_points[0], (0, 255, 255), 3)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(state["points"]) < 2:
                print("还没有点够两个点，请先点击 origin 和 top-left。")
                continue
            cv2.destroyWindow(window_name)
            return state["points"][0], state["points"][1]
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("你取消了两点定框。")


def select_three_points_by_click(image):
    max_display_width = 1200
    max_display_height = 900
    height, width = image.shape[:2]
    scale = min(max_display_width / width, max_display_height / height, 1.0)
    display_size = (int(width * scale), int(height * scale))
    display_base = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
    state = {"points": []}
    window_name = "click 3 frame points, then press Enter"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point = (int(round(x / scale)), int(round(y / scale)))
            if len(state["points"]) >= 3:
                state["points"] = []
            state["points"].append(point)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print()
    print("请依次点击 3 个点：右下原点(0,0)、左下角(30,0)、右上角(0,30)。")
    print("3 个点点击完成后按 Enter 确认；点错时继续点击会重新开始；按 Esc 取消。")

    labels = ("origin (0,0)", "x end (30,0)", "y end (0,30)")
    colors = ((255, 0, 255), (0, 255, 255), (255, 255, 0))
    while True:
        display = display_base.copy()
        cv2.putText(
            display,
            "1) origin  2) left-bottom  3) right-top  Enter=OK",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        display_points = [
            (int(round(point[0] * scale)), int(round(point[1] * scale)))
            for point in state["points"]
        ]
        for index, point in enumerate(display_points):
            cv2.circle(display, point, 8, colors[index], -1)
            cv2.putText(
                display,
                labels[index],
                (point[0] + 10, point[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colors[index],
                2,
                cv2.LINE_AA,
            )
        if len(display_points) == 3:
            origin, x_end, y_end = display_points
            top_left = (x_end[0] + y_end[0] - origin[0], x_end[1] + y_end[1] - origin[1])
            polygon = np.array([top_left, y_end, origin, x_end], dtype=np.int32)
            cv2.polylines(display, [polygon], True, (0, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            if len(state["points"]) < 3:
                print("还没有点够 3 个点，请先完成三个角点。")
                continue
            cv2.destroyWindow(window_name)
            return state["points"][0], state["points"][1], state["points"][2]
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("你取消了三点定框。")


def bbox_from_origin(origin, mm_per_px, image_shape):
    if PATTERN_SIZE_MM <= 0:
        raise ValueError("PATTERN_SIZE_MM 必须大于 0。")

    height, width = image_shape[:2]
    side_px = PATTERN_SIZE_MM / mm_per_px
    x1 = int(round(origin[0]))
    y1 = int(round(origin[1]))
    x0 = int(round(x1 - side_px))
    y0 = int(round(y1 - side_px))

    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("根据原点和 30 mm 尺寸生成的矩形框不合理，请重新点击原点。")
    return (x0, y0, x1, y1)


def bbox_from_two_corners(origin, top_left, image_shape):
    height, width = image_shape[:2]
    x0 = int(round(top_left[0]))
    y0 = int(round(top_left[1]))
    x1 = int(round(origin[0]))
    y1 = int(round(origin[1]))

    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("两点生成的矩形框不合理，请重新点击右下原点和左上角。")
    return (x0, y0, x1, y1)


def choose_measurement_bbox(image, mask, mm_per_px):
    mode = COORDINATE_MODE.strip().lower()
    if mode == "auto_bbox":
        return estimate_frame_bbox(mask), "auto_bbox", None

    if mode == "manual_two_corners":
        if MANUAL_ORIGIN_PX is None or MANUAL_TOP_LEFT_PX is None:
            raise ValueError("COORDINATE_MODE='manual_two_corners' 时，需要填写 MANUAL_ORIGIN_PX 和 MANUAL_TOP_LEFT_PX。")
        origin = MANUAL_ORIGIN_PX
        bbox = bbox_from_two_corners(MANUAL_ORIGIN_PX, MANUAL_TOP_LEFT_PX, image.shape)
        return bbox, mode, origin
    elif mode == "click_two_corners":
        origin, top_left = select_two_corners_by_click(image)
        bbox = bbox_from_two_corners(origin, top_left, image.shape)
        return bbox, mode, origin
    elif mode == "manual_origin_30mm":
        if MANUAL_ORIGIN_PX is None:
            raise ValueError("COORDINATE_MODE='manual_origin_30mm' 时，需要填写 MANUAL_ORIGIN_PX。")
        origin = MANUAL_ORIGIN_PX
    elif mode == "click_origin_30mm":
        origin = select_origin_by_click(image)
    else:
        raise ValueError("COORDINATE_MODE 只能是 click_two_corners、manual_two_corners、click_origin_30mm、manual_origin_30mm 或 auto_bbox。")

    bbox = bbox_from_origin(origin, mm_per_px, image.shape)
    return bbox, mode, origin


def measure_rectangle_widths(image, mask, image_path, mm_per_px, selected_numbers):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("打印线 mask 为空。")

    line_bbox, coordinate_mode, origin = choose_measurement_bbox(image, mask, mm_per_px)
    samples, stable_segments, used_fallback = sample_selected_segments(
        mask, line_bbox, selected_numbers, mm_per_px
    )
    if not samples:
        raise RuntimeError("在你选择的段内没有找到可测量的线宽点，请换一个段编号或检查分割效果。")

    rows = []
    for index, sample in enumerate(samples, start=1):
        rows.append(
            {
                "image": str(image_path),
                "sample_id": index,
                "selected_segment": sample.get("segment_number", ""),
                "segment_name": sample.get("segment_name", ""),
                "segment_coord_start": sample.get("segment_coord_start", ""),
                "segment_coord_end": sample.get("segment_coord_end", ""),
                "side": sample["side"],
                "x": sample["x"],
                "y": sample["y"],
                "width_px": sample["width_px"],
                "width_mm": sample["width_px"] * mm_per_px,
                "stable_segment_start": sample.get("stable_segment_start", ""),
                "stable_segment_end": sample.get("stable_segment_end", ""),
                "coordinate_mode": coordinate_mode,
                "origin_x": "" if origin is None else int(origin[0]),
                "origin_y": "" if origin is None else int(origin[1]),
                "kept": True,
                "reject_reason": "",
            }
        )
    return rows, line_bbox, stable_segments, used_fallback


def measure_rectangle_widths_three_points(
    image, mask, image_path, mm_per_px, selected_numbers, origin, x_axis_end, y_axis_end
):
    """Rectify a rotated or sheared 30 mm frame before sampling its line widths."""
    if PATTERN_SIZE_MM <= 0:
        raise ValueError("PATTERN_SIZE_MM must be greater than zero.")

    side_px = max(80, int(round(PATTERN_SIZE_MM / mm_per_px)))
    source = np.float32([origin, x_axis_end, y_axis_end])
    destination = np.float32(
        [(side_px, side_px), (0, side_px), (side_px, 0)]
    )
    transform = cv2.getAffineTransform(source, destination)
    rectified_image = cv2.warpAffine(
        image,
        transform,
        (side_px + 1, side_px + 1),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rectified_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        transform,
        (side_px + 1, side_px + 1),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    rectified_mask = clean_mask(rectified_mask)

    line_bbox = (0, 0, side_px, side_px)
    samples, stable_segments, used_fallback = sample_selected_segments(
        rectified_mask, line_bbox, selected_numbers, mm_per_px
    )
    if not samples:
        raise RuntimeError("No measurable line-width points were found in the selected segments.")

    rows = []
    for index, sample in enumerate(samples, start=1):
        rows.append(
            {
                "image": str(image_path),
                "sample_id": index,
                "selected_segment": sample.get("segment_number", ""),
                "segment_name": sample.get("segment_name", ""),
                "segment_coord_start": sample.get("segment_coord_start", ""),
                "segment_coord_end": sample.get("segment_coord_end", ""),
                "side": sample["side"],
                "x": sample["x"],
                "y": sample["y"],
                "width_px": sample["width_px"],
                "width_mm": sample["width_px"] * mm_per_px,
                "stable_segment_start": sample.get("stable_segment_start", ""),
                "stable_segment_end": sample.get("stable_segment_end", ""),
                "coordinate_mode": "three_point_rectified",
                "origin_x": int(origin[0]),
                "origin_y": int(origin[1]),
                "kept": True,
                "reject_reason": "",
            }
        )
    inverse_transform = cv2.invertAffineTransform(transform)
    return rows, line_bbox, stable_segments, used_fallback, rectified_image, inverse_transform


def draw_three_point_measurement(image, origin, x_axis_end, y_axis_end, inverse_transform, rows):
    """Draw the affine frame and rectified sample widths back onto the source image."""
    annotated = image.copy()
    top_left = (
        int(x_axis_end[0] + y_axis_end[0] - origin[0]),
        int(x_axis_end[1] + y_axis_end[1] - origin[1]),
    )
    polygon = np.array([top_left, y_axis_end, origin, x_axis_end], dtype=np.int32)
    cv2.polylines(annotated, [polygon], True, (0, 255, 255), 3, cv2.LINE_AA)

    points_and_labels = (
        (origin, "1 origin"),
        (x_axis_end, "2 x-end"),
        (y_axis_end, "3 y-end"),
    )
    for point, label in points_and_labels:
        point = (int(point[0]), int(point[1]))
        cv2.circle(annotated, point, 7, (255, 0, 255), -1)
        cv2.putText(
            annotated,
            label,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for index, row in enumerate(rows, start=1):
        x = float(row["x"])
        y = float(row["y"])
        half = max(4.0, float(row["width_px"]) / 2.0)
        if row["side"] in ("top", "bottom"):
            rectified_points = np.float32([[[x, y - half]], [[x, y + half]], [[x, y]]])
        else:
            rectified_points = np.float32([[[x - half, y]], [[x + half, y]], [[x, y]]])
        source_points = cv2.transform(rectified_points, inverse_transform).reshape(-1, 2)
        p1 = tuple(np.round(source_points[0]).astype(int))
        p2 = tuple(np.round(source_points[1]).astype(int))
        center = tuple(np.round(source_points[2]).astype(int))
        color = (0, 180, 0) if row.get("kept") else (0, 0, 255)
        cv2.line(annotated, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, center, 4, color, -1)
        cv2.putText(
            annotated,
            str(index),
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def validate_width_consistency(rows):
    kept_values = sorted(float(row["width_mm"]) for row in rows if row.get("kept"))
    if len(kept_values) < MIN_VALID_SAMPLES:
        raise RuntimeError(
            f"Only {len(kept_values)} valid samples remain; at least {MIN_VALID_SAMPLES} are required."
        )
    median = float(np.median(kept_values))
    relative_range = (kept_values[-1] - kept_values[0]) / max(median, 1e-9)
    if relative_range > MAX_WIDTH_RANGE_RATIO:
        raise RuntimeError(
            "The selected points are not consistent enough "
            f"(range/median={relative_range:.2f}). Re-click the frame or choose a different segment."
        )
    return {"relative_range": relative_range, "n_kept": len(kept_values)}


# ===================== 输出与标注 =====================

def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image",
            "sample_id",
            "selected_segment",
            "segment_name",
            "segment_coord_start",
            "segment_coord_end",
            "side",
            "x",
            "y",
            "width_px",
            "width_mm",
            "stable_segment_start",
            "stable_segment_end",
            "coordinate_mode",
            "origin_x",
            "origin_y",
            "kept",
            "reject_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(preferred_dir):
    preferred_dir = Path(preferred_dir)
    try:
        preferred_dir.mkdir(parents=True, exist_ok=True)
        return preferred_dir
    except PermissionError:
        fallback = Path(__file__).resolve().parent / "measurement_outputs"
        fallback.mkdir(parents=True, exist_ok=True)
        print(f"无法写入指定输出文件夹，已临时改为: {fallback}")
        return fallback


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


def draw_overall_line_bbox(image, component_bbox):
    annotated = image.copy()
    x, y, w, h = component_bbox
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(
        annotated,
        "detected line area",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_segment_map(image, bbox, selected_numbers):
    annotated = image.copy()
    segment_defs = build_segment_defs(bbox)
    for number, segment in segment_defs.items():
        color = (0, 220, 255) if number in selected_numbers else (180, 180, 180)
        thickness = 4 if number in selected_numbers else 2
        cv2.line(annotated, segment["p1"], segment["p2"], color, thickness)
        mx = int(round((segment["p1"][0] + segment["p2"][0]) / 2.0))
        my = int(round((segment["p1"][1] + segment["p2"][1]) / 2.0))
        cv2.putText(
            annotated,
            str(number),
            (mx + 4, my - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    x0, y0, x1, y1 = bbox
    cv2.circle(annotated, (x1, y1), 7, (255, 0, 255), -1)
    cv2.putText(
        annotated,
        "origin",
        (x1 + 8, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def draw_stable_segments(image, stable_segments, bbox):
    annotated = image.copy()
    if not stable_segments:
        return annotated

    x0, y0, x1, y1 = bbox
    for index, segment in enumerate(stable_segments, start=1):
        side = segment["side"]
        start = int(segment["stable_start"])
        end = int(segment["stable_end"])
        if side == "top":
            p1, p2 = (start, y0), (end, y0)
        elif side == "bottom":
            p1, p2 = (start, y1), (end, y1)
        elif side == "left":
            p1, p2 = (x0, start), (x0, end)
        else:
            p1, p2 = (x1, start), (x1, end)

        cv2.line(annotated, p1, p2, (0, 150, 255), 5)
        cv2.putText(
            annotated,
            f"stable {segment['number']}",
            p1,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def draw_width_samples(image, rows):
    annotated = image.copy()
    for index, row in enumerate(rows, start=1):
        color = (0, 180, 0) if row["kept"] else (0, 0, 255)
        x = int(row["x"])
        y = int(row["y"])
        half = max(4, int(round(row["width_px"] / 2)))
        if row["side"] in ("top", "bottom"):
            p1, p2 = (x, y - half), (x, y + half)
        else:
            p1, p2 = (x - half, y), (x + half, y)
        cv2.line(annotated, p1, p2, color, 2)
        cv2.circle(annotated, (x, y), 4, color, -1)
        cv2.putText(
            annotated,
            str(index),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def main():
    selected_numbers = parse_selected_segments(SELECTED_SEGMENTS)

    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError("请先修改 IMAGE_PATH 为真实图片路径。")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    marker = detect_aruco_marker(image)
    mm_per_px = MARKER_SIZE_MM / marker["side_px"]
    image_stem = image_path.stem

    output_dir = prepare_output_dir(OUTPUT_DIR)
    output_csv = output_dir / f"result_{image_stem}.csv"
    annotated_image = output_dir / f"annotated_{image_stem}.png"
    line_mask_image = output_dir / f"line_mask_{image_stem}.png"

    line_mask, line_component_bbox, chosen_line_mode = detect_line_mask(
        image, marker["bbox"], marker["side_px"]
    )
    rows, line_bbox, stable_segments, used_fallback = measure_rectangle_widths(
        image, line_mask, image_path, mm_per_px, selected_numbers
    )
    rows = reject_outliers(rows, MIN_WIDTH_MM, MAX_WIDTH_MM)
    stats = summarize(rows)

    write_csv(rows, output_csv)
    cv2.imwrite(str(line_mask_image), (line_mask.astype(np.uint8) * 255))

    annotated = draw_overall_line_bbox(image, line_component_bbox)
    annotated = draw_segment_map(annotated, line_bbox, selected_numbers)
    annotated = draw_stable_segments(annotated, stable_segments, line_bbox)
    annotated = draw_width_samples(annotated, rows)
    annotated = draw_aruco_annotation(annotated, marker)
    cv2.imwrite(str(annotated_image), annotated)

    print("Done.")
    print(f"Image: {image_path}")
    print(f"Selected segments: {selected_numbers}")
    print(f"ArUco id: {marker['id']}")
    print(f"Marker bbox: {marker['bbox']}")
    print(f"Marker side: {marker['side_px']:.2f} px")
    print(f"Marker size: {MARKER_SIZE_MM:.4f} mm")
    print(f"Scale: {mm_per_px:.8f} mm/pixel")
    print(f"Line mode: {chosen_line_mode}")
    print(f"Coordinate mode: {COORDINATE_MODE}")
    print(f"Detected line area: {line_component_bbox}")
    print(f"Measurement bbox: {line_bbox}")
    print(f"Stable segments: {len(stable_segments)}")
    print(f"Used fallback sampling: {used_fallback}")
    print(f"Measured samples: {len(rows)}")
    print(f"Kept samples: {stats['n_kept']}")
    print(f"Mean width: {stats['mean_mm']:.6f} mm")
    print(f"Median width: {stats['median_mm']:.6f} mm")
    print(f"CSV: {output_csv}")
    print(f"Annotated image: {annotated_image}")
    print(f"Line mask: {line_mask_image}")


if __name__ == "__main__":
    main()
