from pathlib import Path
import importlib.util
import math
import random
import sys
import traceback

import cv2


# ===================== Main settings =====================

IMAGE_DIR = Path(r"D:\bio-print\database\data\raw\images\photos")
EXCEL_PATH = Path(r"D:\bio-print\database\data\raw\metadata\bioprint_metadata_template_v2.xlsx")

# Save one visual quality-control image for each measured photo.
# No masks, CSV files, or Excel backup files are created.
SAVE_ANNOTATED_IMAGES = True
ANNOTATED_OUTPUT_DIR = Path(r"D:\bio-print\database\data\raw\images\annotated_pictures")
OUTPUT_DIR = None

# The single-image ArUco measurement script must be in the same folder as this batch script.
MEASURE_SCRIPT_PATH = Path(__file__).with_name("线宽测量测试_ArUco版.py")

# Your current default choice. Each image will still ask you to edit this.
DEFAULT_SEGMENTS = "2,3,4,5,6,7"

# ArUco black square outer edge size in mm.
MARKER_SIZE_MM = 19.2

# Marker dictionary. 25.jpg is detected as APRILTAG_36H11 ID 0.
MARKER_DICTIONARY_NAME = "DICT_APRILTAG_36H11"

# Printed rectangle size in mm.
PATTERN_SIZE_MM = 30.0

# Width sample count per image.
SAMPLES = 10

# Keep "auto" first. If needed, change to "color_excess" or "yellow_green".
LINE_MODE = "auto"

# Optional width filters, in mm.
MIN_WIDTH_MM = None
MAX_WIDTH_MM = None

# When measuring one edge, only search this far from that edge.
EDGE_SEARCH_BAND_MM = 6.0

# Excel settings.
SHEET_NAME = None  # None means active sheet.
ID_COLUMN_NAME = "编号"
SAVE_AFTER_EACH_IMAGE = True

RESULT_COLUMNS = {
    "mean": "线宽平均值(mm)",
    "median": "线宽中位数(mm)",
    "max": "线宽最大值(mm)",
    "min": "线宽最小值(mm)",
    "random": "线宽随机测点值(mm)",
}


# ===================== Dialog helpers =====================

def ask_text(title, prompt, default_value=""):
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        value = simpledialog.askstring(title, prompt, initialvalue=default_value)
        root.destroy()
        return value
    except Exception:
        print()
        print(prompt)
        if default_value:
            print(f"默认值: {default_value}")
        value = input("> ").strip()
        return value if value else default_value


def parse_image_range(text):
    text = str(text).strip().replace("，", ",")
    if "-" in text:
        start_text, end_text = text.split("-", 1)
        start = int(start_text.strip())
        end = int(end_text.strip())
        if end < start:
            raise ValueError("图片范围结束编号不能小于开始编号。")
        return list(range(start, end + 1))

    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("没有输入图片编号。")
    return values


def find_image_path(image_id):
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        path = IMAGE_DIR / f"{image_id}{suffix}"
        if path.exists():
            return path
    return None


# ===================== Measurement module =====================

def load_measure_module():
    if not MEASURE_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"找不到单张测量脚本: {MEASURE_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("aruco_line_width_measure", MEASURE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aruco_line_width_measure"] = module
    spec.loader.exec_module(module)

    module.MARKER_SIZE_MM = MARKER_SIZE_MM
    module.MARKER_DICTIONARY_NAME = MARKER_DICTIONARY_NAME
    module.OUTPUT_DIR = OUTPUT_DIR
    module.SAMPLES = SAMPLES
    module.LINE_MODE = LINE_MODE
    module.LINE_BBOX = None
    module.MIN_WIDTH_MM = MIN_WIDTH_MM
    module.MAX_WIDTH_MM = MAX_WIDTH_MM
    module.COORDINATE_MODE = "manual_two_corners"
    module.PATTERN_SIZE_MM = PATTERN_SIZE_MM
    module.EDGE_SEARCH_BAND_MM = EDGE_SEARCH_BAND_MM
    return module


def measure_one_image(measure, image_id, image_path, segment_text):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    print()
    print(f"正在测量图片 {image_id}: {image_path}")

    marker = measure.detect_aruco_marker(image)
    mm_per_px = MARKER_SIZE_MM / marker["side_px"]

    line_mask, line_component_bbox, chosen_line_mode = measure.detect_line_mask(
        image, marker["bbox"], marker["side_px"]
    )

    origin, top_left = measure.select_two_corners_by_click(image)
    measure.MANUAL_ORIGIN_PX = origin
    measure.MANUAL_TOP_LEFT_PX = top_left

    selected_numbers = measure.parse_selected_segments(segment_text)
    rows, line_bbox, stable_segments, used_fallback = measure.measure_rectangle_widths(
        image, line_mask, image_path, mm_per_px, selected_numbers
    )
    rows = measure.reject_outliers(rows, MIN_WIDTH_MM, MAX_WIDTH_MM)
    stats = measure.summarize(rows)

    return {
        "image_id": image_id,
        "segments": ",".join(str(v) for v in selected_numbers),
        "stats": stats,
        "rows": rows,
        "total_samples": len(rows),
        "status": "完成",
        "chosen_line_mode": chosen_line_mode,
        "used_fallback": used_fallback,
        "origin": origin,
    }


# ===================== Excel helpers =====================

def load_workbook_tools():
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("缺少 openpyxl。请先运行: pip install openpyxl") from exc
    return load_workbook


def normalize_header(value):
    return "" if value is None else str(value).strip()


def find_header_row_and_columns(ws):
    for row in range(1, min(ws.max_row, 30) + 1):
        headers = {}
        for col in range(1, ws.max_column + 1):
            header = normalize_header(ws.cell(row=row, column=col).value)
            if header:
                headers[header] = col
        if ID_COLUMN_NAME in headers:
            return row, headers
    raise RuntimeError(f"Excel 前 30 行里没有找到列名: {ID_COLUMN_NAME}")


def ensure_result_columns(ws, header_row, headers):
    next_col = ws.max_column + 1
    for header in RESULT_COLUMNS.values():
        if header not in headers:
            ws.cell(row=header_row, column=next_col).value = header
            headers[header] = next_col
            next_col += 1
    return headers


def id_matches(cell_value, image_id):
    if cell_value is None:
        return False
    text = str(cell_value).strip()
    if text == str(image_id):
        return True
    try:
        return int(float(text)) == int(image_id)
    except ValueError:
        return False


def build_id_to_row(ws, header_row, id_col):
    result = {}
    for row in range(header_row + 1, ws.max_row + 1):
        value = ws.cell(row=row, column=id_col).value
        if value is None:
            continue
        try:
            key = int(float(str(value).strip()))
        except ValueError:
            continue
        result[key] = row
    return result


def write_result_to_excel(ws, headers, row_number, result):
    values = result["excel_values"]
    for header, value in values.items():
        ws.cell(row=row_number, column=headers[header]).value = value


def build_excel_values(rows, image_id):
    """Build five traceable values from the accepted width samples of one image."""
    kept_values = sorted(
        float(row["width_mm"])
        for row in rows
        if row.get("kept") and math.isfinite(float(row["width_mm"]))
    )
    if not kept_values:
        raise RuntimeError("10 个测点筛选后没有保留合理的线宽值，请重新检查选段和检测框。")

    count = len(kept_values)
    middle = count // 2
    median = (
        kept_values[middle]
        if count % 2 == 1
        else (kept_values[middle - 1] + kept_values[middle]) / 2.0
    )

    # Use the image ID as a seed so a repeated run remains reproducible.
    random_value = random.Random(str(image_id)).choice(kept_values)
    return {
        RESULT_COLUMNS["median"]: median,
        RESULT_COLUMNS["mean"]: sum(kept_values) / count,
        RESULT_COLUMNS["max"]: kept_values[-1],
        RESULT_COLUMNS["min"]: kept_values[0],
        RESULT_COLUMNS["random"]: random_value,
    }


def write_failure_to_excel(ws, headers, row_number, message):
    # Only result values are written to Excel. Failures are printed in the terminal.
    return


# ===================== Main flow =====================

def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {IMAGE_DIR}")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {EXCEL_PATH}")

    range_text = ask_text("选择图片范围", "请输入要测量的图片编号范围，例如 25-50，或 25,28,31。")
    image_ids = parse_image_range(range_text)

    load_workbook = load_workbook_tools()
    workbook = load_workbook(EXCEL_PATH)
    ws = workbook[SHEET_NAME] if SHEET_NAME else workbook.active
    header_row, headers = find_header_row_and_columns(ws)
    headers = ensure_result_columns(ws, header_row, headers)
    id_to_row = build_id_to_row(ws, header_row, headers[ID_COLUMN_NAME])

    measure = load_measure_module()

    completed = 0
    failed = 0
    skipped = 0

    for image_id in image_ids:
        row_number = id_to_row.get(int(image_id))
        if row_number is None:
            print(f"跳过 {image_id}: Excel 中找不到编号为 {image_id} 的行。")
            skipped += 1
            continue

        image_path = find_image_path(image_id)
        if image_path is None:
            message = f"找不到图片 {image_id}.jpg"
            print(f"跳过 {image_id}: {message}")
            write_failure_to_excel(ws, headers, row_number, message)
            failed += 1
            continue

        print()
        print(f"图片 {image_id}: 请先点击右下原点，再点击左上角。")
        try:
            print(f"Image {image_id}: click (1) right-bottom, (2) left-bottom, (3) right-top.")
            image_preview = cv2.imread(str(image_path))
            if image_preview is None:
                raise FileNotFoundError(f"无法读取图片: {image_path}")

            # Three known corners rectify a rotated or sheared print frame.
            measure_image = image_preview
            marker = None
            line_mask = None
            line_component_bbox = None
            chosen_line_mode = None
            marker = measure.detect_aruco_marker(measure_image)
            mm_per_px = MARKER_SIZE_MM / marker["side_px"]
            line_mask, line_component_bbox, chosen_line_mode = measure.detect_line_mask(
                measure_image, marker["bbox"], marker["side_px"]
            )
            origin, x_axis_end, y_axis_end = measure.select_three_points_by_click(measure_image)
            measure.MANUAL_ORIGIN_PX = origin
            measure.MANUAL_X_AXIS_PX = x_axis_end
            measure.MANUAL_Y_AXIS_PX = y_axis_end

            segment_text = ask_text(
                f"图片 {image_id} 的测量片段",
                "请输入要测量的片段编号。直接确认使用默认值；输入 skip 跳过；输入 stop 结束。",
                DEFAULT_SEGMENTS,
            )
            if segment_text is None:
                print("用户取消，结束批量测量。")
                break
            segment_text = segment_text.strip()
            if segment_text.lower() in ("stop", "q", "quit", "exit"):
                print("用户选择结束。")
                break
            if segment_text.lower() in ("skip", "s"):
                print(f"跳过图片 {image_id}。")
                skipped += 1
                continue

            selected_numbers = measure.parse_selected_segments(segment_text)
            rows, line_bbox, stable_segments, used_fallback, rectified_image, inverse_transform = (
                measure.measure_rectangle_widths_three_points(
                    measure_image,
                    line_mask,
                    image_path,
                    mm_per_px,
                    selected_numbers,
                    origin,
                    x_axis_end,
                    y_axis_end,
                )
            )
            rows = measure.reject_outliers(rows, MIN_WIDTH_MM, MAX_WIDTH_MM)
            stats = measure.summarize(rows)

            if SAVE_ANNOTATED_IMAGES:
                ANNOTATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                annotated = measure.draw_overall_line_bbox(measure_image, line_component_bbox)
                annotated = measure.draw_three_point_measurement(
                    annotated,
                    origin,
                    x_axis_end,
                    y_axis_end,
                    inverse_transform,
                    rows,
                )
                annotated = measure.draw_aruco_annotation(annotated, marker)
                annotated_path = ANNOTATED_OUTPUT_DIR / f"annotated_{image_path.stem}.png"
                if not cv2.imwrite(str(annotated_path), annotated):
                    raise RuntimeError(f"Unable to save annotated image: {annotated_path}")

            consistency = measure.validate_width_consistency(rows)

            result = {
                "image_id": image_id,
                "segments": ",".join(str(v) for v in selected_numbers),
                "stats": stats,
                "rows": rows,
                "excel_values": build_excel_values(rows, image_id),
                "total_samples": len(rows),
                "status": "完成",
                "chosen_line_mode": chosen_line_mode,
                "used_fallback": used_fallback,
                "origin": origin,
                "x_axis_end": x_axis_end,
                "y_axis_end": y_axis_end,
                "relative_range": consistency["relative_range"],
            }
            write_result_to_excel(ws, headers, row_number, result)

            completed += 1

            print(
                f"完成 {image_id}: 平均={stats['mean_mm']:.6f} mm, "
                f"中位数={stats['median_mm']:.6f} mm, 有效点={stats['n_kept']}"
            )

            if SAVE_AFTER_EACH_IMAGE:
                workbook.save(EXCEL_PATH)

        except Exception as exc:
            failed += 1
            short_message = str(exc).splitlines()[0][:120]
            print(f"图片 {image_id} 失败: {short_message}")
            traceback.print_exc()
            write_failure_to_excel(ws, headers, row_number, short_message)
            if SAVE_AFTER_EACH_IMAGE:
                workbook.save(EXCEL_PATH)

    workbook.save(EXCEL_PATH)
    print()
    print("批量测量结束。")
    print(f"完成: {completed}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print(f"Excel: {EXCEL_PATH}")


if __name__ == "__main__":
    main()
