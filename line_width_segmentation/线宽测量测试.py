from pathlib import Path

import cv2
import numpy as np

from line_width_measure import (
    detect_reference_bbox,
    draw_annotation,
    measure_at,
    sample_positions,
    threshold_line,
    allocate_samples,
    reject_outliers,
    summarize,
)


# Change these two values before running in VS Code.
IMAGE_PATH = r"D:\bio-print\database\data\raw\images\test.jpg"
KNOWN_WIDTH_MM = 19.2

# These defaults fit black-background, yellow/bright printed lines.
REF_AXIS = "width"  # use "height" if the known size is the reference object's height
MODE = "bright"
SAMPLES = 10
MIN_WIDTH_MM = None
MAX_WIDTH_MM = None


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "result.csv"
ANNOTATED_IMAGE = OUTPUT_DIR / "annotated.png"


def write_csv(rows, csv_path):
    import csv

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


def main():
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        raise FileNotFoundError(
            "Please edit IMAGE_PATH at the top of this file before running it."
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]
    ref_detection = detect_reference_bbox(
        image,
        mode=MODE,
        line_bbox=None,
        min_area=max(100, int(img_w * img_h * 0.0002)),
    )
    ref_bbox = ref_detection["bbox"]
    ref_px = ref_bbox[2] if REF_AXIS == "width" else ref_bbox[3]
    mm_per_px = KNOWN_WIDTH_MM / float(ref_px)

    mask, chosen_mode = threshold_line(image, ref_bbox=ref_bbox, mode=MODE)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("The printed line mask is empty.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = (x0, y0, x1, y1)

    horizontal_len = max(1, x1 - x0 + 1)
    vertical_len = max(1, y1 - y0 + 1)
    counts = allocate_samples(SAMPLES, horizontal_len, vertical_len)
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

    rows = reject_outliers(rows, MIN_WIDTH_MM, MAX_WIDTH_MM)
    stats = summarize(rows)
    write_csv(rows, OUTPUT_CSV)

    annotated = draw_annotation(image, rows, ref_bbox=ref_bbox, line_bbox=bbox)
    cv2.imwrite(str(ANNOTATED_IMAGE), annotated)

    print("Done.")
    print(f"Detected reference bbox: {ref_bbox}")
    print(f"Scale: {mm_per_px:.8f} mm/pixel")
    print(f"Segmentation mode: {chosen_mode}")
    print(f"Measured samples: {len(rows)}")
    print(f"Kept samples: {stats['n_kept']}")
    print(f"Mean width: {stats['mean_mm']:.6f} mm")
    print(f"Median width: {stats['median_mm']:.6f} mm")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Annotated image: {ANNOTATED_IMAGE}")


if __name__ == "__main__":
    main()
