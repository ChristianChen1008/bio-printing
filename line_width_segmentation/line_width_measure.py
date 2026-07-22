import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_bbox(text):
    values = [int(v.strip()) for v in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h")
    x, y, w, h = values
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("bbox width and height must be positive")
    return x, y, w, h


def clamp_bbox(bbox, width, height):
    x, y, w, h = bbox
    x0 = max(0, min(width - 1, x))
    y0 = max(0, min(height - 1, y))
    x1 = max(1, min(width, x + w))
    y1 = max(1, min(height, y + h))
    return x0, y0, x1 - x0, y1 - y0


def clean_mask(mask):
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    return mask.astype(bool)


def foreground_candidates(image, mode="auto"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_masks = {
        "dark": gray < otsu_value,
        "bright": gray > otsu_value,
    }
    for name, raw in raw_masks.items():
        if mode != "auto" and mode != name:
            continue
        yield name, clean_mask(raw)


def bbox_overlap_ratio(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    return intersection / float(max(1, aw * ah))


def detect_reference_bbox(image, mode="auto", line_bbox=None, min_area=100):
    height, width = image.shape[:2]
    image_area = height * width
    line_bbox = clamp_bbox(line_bbox, width, height) if line_bbox else None
    best = None

    for mask_name, mask in foreground_candidates(image, mode):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > image_area * 0.4:
                continue
            if w < 8 or h < 8:
                continue

            bbox = (x, y, w, h)
            if line_bbox and bbox_overlap_ratio(bbox, line_bbox) > 0.25:
                continue

            fill_ratio = area / float(w * h)
            aspect = max(w / float(h), h / float(w))
            if fill_ratio < 0.45 or aspect > 12:
                continue

            score = area * fill_ratio / math.sqrt(aspect)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "bbox": bbox,
                    "mode": mask_name,
                    "area": area,
                    "fill_ratio": fill_ratio,
                    "aspect": aspect,
                }

    if best is None:
        raise RuntimeError(
            "Could not auto-detect the reference object. Try adding --ref-bbox, "
            "--line-bbox, or --ref-mode bright/dark."
        )
    return best


def threshold_line(image, line_bbox=None, ref_bbox=None, mode="auto"):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu_value, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray < otsu_value
    bright = gray > otsu_value

    region = np.zeros((height, width), dtype=bool)
    if line_bbox:
        x, y, w, h = clamp_bbox(line_bbox, width, height)
        region[y : y + h, x : x + w] = True
    else:
        region[:, :] = True

    if ref_bbox:
        x, y, w, h = clamp_bbox(ref_bbox, width, height)
        region[y : y + h, x : x + w] = False

    candidates = []
    for name, raw in (("dark", dark), ("bright", bright)):
        if mode != "auto" and mode != name:
            continue
        mask = clean_mask(raw & region)
        ratio = float(mask.sum()) / max(1, int(region.sum()))
        if 0.001 <= ratio <= 0.55:
            candidates.append((ratio, name, mask))

    if not candidates:
        raise RuntimeError(
            "Could not segment the printed line. Try --line-bbox and --mode dark/bright."
        )

    candidates.sort(key=lambda item: item[0])
    _, chosen_name, mask = candidates[0] if mode == "auto" else candidates[-1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        raise RuntimeError("No connected printed-line component was found.")

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    component = labels == largest_label
    return component, chosen_name


def contiguous_runs(indices):
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return runs


def allocate_samples(total, horizontal_len, vertical_len):
    if total < 4:
        raise ValueError("--samples must be at least 4")

    edge_lengths = np.array(
        [horizontal_len, vertical_len, horizontal_len, vertical_len], dtype=float
    )
    counts = np.ones(4, dtype=int)
    remaining = total - 4
    if remaining > 0:
        weights = edge_lengths / edge_lengths.sum()
        extras = np.floor(weights * remaining).astype(int)
        counts += extras
        while counts.sum() < total:
            residual = weights * remaining - extras
            idx = int(np.argmax(residual))
            counts[idx] += 1
            extras[idx] += 1
    return {
        "top": int(counts[0]),
        "right": int(counts[1]),
        "bottom": int(counts[2]),
        "left": int(counts[3]),
    }


def sample_positions(start, end, count, margin):
    usable_start = start + margin
    usable_end = end - margin
    if usable_end <= usable_start:
        usable_start, usable_end = start, end
    return np.linspace(usable_start, usable_end, count + 2, dtype=int)[1:-1]


def measure_at(mask, side, x, y, bbox):
    x0, y0, x1, y1 = bbox
    if side in ("top", "bottom"):
        ys = np.flatnonzero(mask[y0 : y1 + 1, x]) + y0
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

    xs = np.flatnonzero(mask[y, x0 : x1 + 1]) + x0
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
    kept = []
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
        kept.append(row)

    candidate_values = np.array([r["width_mm"] for r in kept if r["kept"]], dtype=float)
    if len(candidate_values) >= 5:
        median = float(np.median(candidate_values))
        mad = float(np.median(np.abs(candidate_values - median)))
        if mad > 1e-9:
            for row in kept:
                if not row["kept"]:
                    continue
                robust_z = 0.6745 * abs(row["width_mm"] - median) / mad
                if robust_z > 3.5:
                    row["kept"] = False
                    row["reject_reason"] = "outlier"
    return kept


def draw_annotation(image, rows, ref_bbox=None, line_bbox=None):
    annotated = image.copy()
    if ref_bbox:
        x, y, w, h = ref_bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 180, 0), 2)
        cv2.putText(
            annotated,
            "reference",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
    if line_bbox:
        x0, y0, x1, y1 = line_bbox
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (120, 120, 255), 1)

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


def main():
    parser = argparse.ArgumentParser(
        description="Measure printed line width from a rectangular printed pattern."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--known-width-mm", type=float, required=True, help="Real reference width.")
    parser.add_argument(
        "--ref-bbox",
        type=parse_bbox,
        help="Reference object box in pixels: x,y,w,h. If omitted, auto-detection is used.",
    )
    parser.add_argument(
        "--ref-axis",
        choices=("width", "height"),
        default="width",
        help="Which side of ref-bbox equals --known-width-mm.",
    )
    parser.add_argument(
        "--ref-mode",
        choices=("auto", "dark", "bright"),
        help="Auto-detect the reference as a dark or bright object. Defaults to --mode.",
    )
    parser.add_argument("--line-bbox", type=parse_bbox, help="Optional printed-pattern box: x,y,w,h.")
    parser.add_argument(
        "--mode",
        choices=("auto", "dark", "bright"),
        default="auto",
        help="Use dark lines, bright lines, or automatic thresholding.",
    )
    parser.add_argument("--samples", type=int, default=10, help="Number of width samples.")
    parser.add_argument("--min-width-mm", type=float, help="Reject widths below this value.")
    parser.add_argument("--max-width-mm", type=float, help="Reject widths above this value.")
    parser.add_argument("--output-csv", default="line_width_results.csv")
    parser.add_argument("--annotated-image", default="line_width_annotated.png")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]
    if args.ref_bbox:
        ref_bbox = clamp_bbox(args.ref_bbox, img_w, img_h)
        ref_detection = None
    else:
        ref_detection = detect_reference_bbox(
            image,
            mode=args.ref_mode or args.mode,
            line_bbox=args.line_bbox,
            min_area=max(100, int(img_w * img_h * 0.0002)),
        )
        ref_bbox = ref_detection["bbox"]

    ref_px = ref_bbox[2] if args.ref_axis == "width" else ref_bbox[3]
    mm_per_px = args.known_width_mm / float(ref_px)

    mask, chosen_mode = threshold_line(
        image, line_bbox=args.line_bbox, ref_bbox=ref_bbox, mode=args.mode
    )
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("The printed line mask is empty.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = (x0, y0, x1, y1)

    horizontal_len = max(1, x1 - x0 + 1)
    vertical_len = max(1, y1 - y0 + 1)
    counts = allocate_samples(args.samples, horizontal_len, vertical_len)
    margin_x = max(2, int(horizontal_len * 0.12))
    margin_y = max(2, int(vertical_len * 0.12))

    samples = []
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
        width_mm = sample["width_px"] * mm_per_px
        rows.append(
            {
                "image": str(image_path),
                "sample_id": index,
                "side": sample["side"],
                "x": sample["x"],
                "y": sample["y"],
                "width_px": sample["width_px"],
                "width_mm": width_mm,
                "kept": True,
                "reject_reason": "",
            }
        )

    rows = reject_outliers(rows, args.min_width_mm, args.max_width_mm)
    stats = summarize(rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
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

    annotated = draw_annotation(image, rows, ref_bbox=ref_bbox, line_bbox=bbox)
    annotated_path = Path(args.annotated_image)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_path), annotated)

    print(f"segmentation_mode={chosen_mode}")
    if ref_detection:
        x, y, w, h = ref_bbox
        print(f"auto_ref_bbox={x},{y},{w},{h}")
        print(f"auto_ref_mode={ref_detection['mode']}")
    print(f"scale_mm_per_px={mm_per_px:.8f}")
    print(f"requested_samples={args.samples}")
    print(f"measured_samples={len(rows)}")
    print(f"kept_samples={stats['n_kept']}")
    print(f"mean_width_mm={stats['mean_mm']:.6f}")
    print(f"median_width_mm={stats['median_mm']:.6f}")
    print(f"std_width_mm={stats['std_mm']:.6f}")
    print(f"csv={output_csv}")
    print(f"annotated_image={annotated_path}")


if __name__ == "__main__":
    main()
