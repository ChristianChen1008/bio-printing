import argparse
import csv
import heapq
import inspect
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--bin-size", type=float, default=100.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-profile-csv", default=None)
    parser.add_argument("--output-bin-csv", default=None)
    parser.add_argument("--output-clean-mask", default=None)
    parser.add_argument("--output-overlay", default=None)
    parser.add_argument("--output-heatmap", default=None)
    return parser.parse_args()


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def remove_small_regions(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    signature = inspect.signature(remove_small_objects)
    if "max_size" in signature.parameters:
        return remove_small_objects(binary_mask, max_size=max(0, min_area - 1))
    return remove_small_objects(binary_mask, min_size=min_area)


def neighbors_8(point: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    y, x = point
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            weight = math.sqrt(2.0) if dy and dx else 1.0
            result.append(((y + dy, x + dx), weight))
    return result


def dijkstra_distances(
    points: set[tuple[int, int]],
    start: tuple[int, int],
) -> dict[tuple[int, int], float]:
    distances = {start: 0.0}
    queue = [(0.0, start)]

    while queue:
        distance, point = heapq.heappop(queue)
        if distance > distances[point]:
            continue

        for neighbor, weight in neighbors_8(point):
            if neighbor not in points:
                continue
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))

    return distances


def choose_component_start(points: set[tuple[int, int]]) -> tuple[int, int]:
    degrees = {}
    for point in points:
        degrees[point] = sum(1 for neighbor, _ in neighbors_8(point) if neighbor in points)

    endpoints = [point for point, degree in degrees.items() if degree == 1]
    candidates = endpoints or list(points)

    # Pick one end of the longest path approximation so distance increases along
    # the main stroke instead of depending on image scan order.
    first = min(candidates)
    first_distances = dijkstra_distances(points, first)
    farthest = max(candidates, key=lambda point: first_distances.get(point, -1.0))
    second_distances = dijkstra_distances(points, farthest)
    return max(candidates, key=lambda point: second_distances.get(point, -1.0))


def profile_dtype() -> np.dtype:
    return np.dtype([
        ("component_id", "i4"),
        ("point_index", "i4"),
        ("distance_px", "f8"),
        ("distance_real", "f8"),
        ("x", "i4"),
        ("y", "i4"),
        ("width_px", "f8"),
        ("width_real", "f8"),
    ])


def bins_dtype() -> np.dtype:
    return np.dtype([
        ("component_id", "i4"),
        ("start_real", "f8"),
        ("end_real", "f8"),
        ("count", "i4"),
        ("mean_width_real", "f8"),
        ("median_width_real", "f8"),
        ("min_width_real", "f8"),
        ("max_width_real", "f8"),
        ("std_width_real", "f8"),
    ])


def build_width_profile(
    skeleton: np.ndarray,
    distance: np.ndarray,
    pixel_size: float,
) -> np.ndarray:
    component_count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    rows = []

    for component_id in range(1, component_count):
        ys, xs = np.where(labels == component_id)
        points = set(zip(ys.tolist(), xs.tolist()))
        if not points:
            continue

        start = choose_component_start(points)
        distances = dijkstra_distances(points, start)
        ordered_points = sorted(points, key=lambda point: (distances.get(point, 0.0), point[0], point[1]))

        for point_index, (y, x) in enumerate(ordered_points):
            distance_px = distances.get((y, x), 0.0)
            width_px = float(distance[y, x] * 2.0)
            if width_px <= 0:
                continue
            rows.append((
                component_id,
                point_index,
                distance_px,
                distance_px * pixel_size,
                x,
                y,
                width_px,
                width_px * pixel_size,
            ))

    return np.array(rows, dtype=profile_dtype())


def build_binned_statistics(profile: np.ndarray, bin_size: float) -> np.ndarray:
    if profile.size == 0:
        return np.array([], dtype=bins_dtype())

    rows = []
    for component_id in np.unique(profile["component_id"]):
        component = profile[profile["component_id"] == component_id]
        max_distance = float(np.max(component["distance_real"]))
        bin_count = max(1, int(math.ceil(max_distance / bin_size)))

        for bin_index in range(bin_count):
            start = bin_index * bin_size
            end = start + bin_size
            if bin_index == bin_count - 1:
                selected = component[
                    (component["distance_real"] >= start)
                    & (component["distance_real"] <= end)
                ]
            else:
                selected = component[
                    (component["distance_real"] >= start)
                    & (component["distance_real"] < end)
                ]
            if selected.size == 0:
                continue

            widths = selected["width_real"]
            rows.append((
                int(component_id),
                start,
                end,
                int(widths.size),
                float(np.mean(widths)),
                float(np.median(widths)),
                float(np.min(widths)),
                float(np.max(widths)),
                float(np.std(widths)),
            ))

    return np.array(rows, dtype=bins_dtype())


def compute_width_analysis(
    binary_mask: np.ndarray,
    pixel_size: float = 1.0,
    bin_size: float = 100.0,
    min_area: int = 20,
) -> dict[str, np.ndarray | dict[str, float]]:
    binary = remove_small_regions(binary_mask.astype(bool), min_area=min_area)
    skeleton = skeletonize(binary)
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    profile = build_width_profile(skeleton, distance, pixel_size)

    if profile.size == 0:
        raise ValueError("No skeleton pixels found. Check whether the mask is empty.")

    return {
        "binary": binary,
        "skeleton": skeleton,
        "profile": profile,
        "bins": build_binned_statistics(profile, bin_size),
        "stats_px": summarize(profile["width_px"]),
        "stats_real": summarize(profile["width_real"]),
    }


def save_overlay(
    mask: np.ndarray,
    skeleton: np.ndarray,
    output_path: str | Path,
) -> None:
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    overlay[mask] = (60, 180, 255)
    overlay[skeleton] = (0, 0, 255)
    cv2.imwrite(str(output_path), overlay)


def save_binary_mask(mask: np.ndarray, output_path: str | Path) -> None:
    cv2.imwrite(str(output_path), mask.astype(np.uint8) * 255)


def save_heatmap(
    mask: np.ndarray,
    profile: np.ndarray,
    output_path: str | Path,
) -> None:
    base = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    base[mask] = (45, 45, 45)

    widths = profile["width_real"]
    min_width = float(np.min(widths))
    max_width = float(np.max(widths))
    denominator = max(max_width - min_width, 1e-7)

    normalized = ((widths - min_width) / denominator * 255).astype(np.uint8)
    colors = cv2.applyColorMap(normalized[:, None], cv2.COLORMAP_TURBO)[:, 0, :]

    for row, color in zip(profile, colors):
        cv2.circle(base, (int(row["x"]), int(row["y"])), 1, color.tolist(), -1)

    cv2.imwrite(str(output_path), base)


def write_structured_csv(rows: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(rows.dtype.names)
        for row in rows:
            writer.writerow([row[name] for name in rows.dtype.names])


def resolve_measure_outputs(
    mask_path: str | Path,
    output_dir: str | Path | None,
    output_csv: str | Path | None,
    output_profile_csv: str | Path | None,
    output_bin_csv: str | Path | None,
    output_overlay: str | Path | None,
    output_heatmap: str | Path | None,
    output_clean_mask: str | Path | None,
) -> dict[str, Path | None]:
    if output_dir is None:
        return {
            "summary_csv": Path(output_csv) if output_csv else None,
            "profile_csv": Path(output_profile_csv) if output_profile_csv else None,
            "bin_csv": Path(output_bin_csv) if output_bin_csv else None,
            "clean_mask": Path(output_clean_mask) if output_clean_mask else None,
            "overlay": Path(output_overlay) if output_overlay else None,
            "heatmap": Path(output_heatmap) if output_heatmap else None,
        }

    base_dir = Path(output_dir)
    mask_path = Path(mask_path)
    try:
        relative_mask = mask_path.resolve().relative_to(base_dir.resolve())
        group_name = relative_mask.parts[0] if len(relative_mask.parts) > 1 else mask_path.stem
    except ValueError:
        group_name = mask_path.stem

    group_dir = base_dir / group_name
    return {
        "summary_csv": Path(output_csv) if output_csv else group_dir / "summary.csv",
        "profile_csv": Path(output_profile_csv) if output_profile_csv else group_dir / "width_profile.csv",
        "bin_csv": Path(output_bin_csv) if output_bin_csv else group_dir / "width_bins.csv",
        "clean_mask": Path(output_clean_mask) if output_clean_mask else group_dir / "clean_mask.png",
        "overlay": Path(output_overlay) if output_overlay else group_dir / "width_overlay.png",
        "heatmap": Path(output_heatmap) if output_heatmap else group_dir / "width_heatmap.png",
    }


def main() -> None:
    args = parse_args()
    mask_gray = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise ValueError(f"Cannot read mask: {args.mask}")

    binary = mask_gray > 127
    analysis = compute_width_analysis(
        binary,
        pixel_size=args.pixel_size,
        bin_size=args.bin_size,
        min_area=args.min_area,
    )
    stats_px = analysis["stats_px"]
    stats_real = analysis["stats_real"]
    profile = analysis["profile"]
    bins = analysis["bins"]
    outputs = resolve_measure_outputs(
        mask_path=args.mask,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        output_profile_csv=args.output_profile_csv,
        output_bin_csv=args.output_bin_csv,
        output_overlay=args.output_overlay,
        output_heatmap=args.output_heatmap,
        output_clean_mask=args.output_clean_mask,
    )

    print("Width statistics")
    print(f"  skeleton points: {int(stats_px['count'])}")
    print(f"  mean:   {stats_px['mean']:.4f} px | {stats_real['mean']:.4f} {args.unit}")
    print(f"  median: {stats_px['median']:.4f} px | {stats_real['median']:.4f} {args.unit}")
    print(f"  min:    {stats_px['min']:.4f} px | {stats_real['min']:.4f} {args.unit}")
    print(f"  max:    {stats_px['max']:.4f} px | {stats_real['max']:.4f} {args.unit}")
    print(f"  std:    {stats_px['std']:.4f} px | {stats_real['std']:.4f} {args.unit}")

    if outputs["summary_csv"]:
        csv_path = outputs["summary_csv"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["metric", "pixel", args.unit])
            for key in ["mean", "median", "min", "max", "std"]:
                writer.writerow([key, stats_px[key], stats_real[key]])
        print(f"Saved CSV: {csv_path}")

    if outputs["profile_csv"]:
        write_structured_csv(profile, outputs["profile_csv"])
        print(f"Saved profile CSV: {outputs['profile_csv']}")

    if outputs["bin_csv"]:
        write_structured_csv(bins, outputs["bin_csv"])
        print(f"Saved binned CSV: {outputs['bin_csv']}")

    if outputs["clean_mask"]:
        clean_mask_path = outputs["clean_mask"]
        clean_mask_path.parent.mkdir(parents=True, exist_ok=True)
        save_binary_mask(analysis["binary"], clean_mask_path)
        print(f"Saved clean mask: {clean_mask_path}")

    if outputs["overlay"]:
        overlay_path = outputs["overlay"]
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        save_overlay(analysis["binary"], analysis["skeleton"], overlay_path)
        print(f"Saved overlay: {overlay_path}")

    if outputs["heatmap"]:
        heatmap_path = outputs["heatmap"]
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        save_heatmap(analysis["binary"], profile, heatmap_path)
        print(f"Saved heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
