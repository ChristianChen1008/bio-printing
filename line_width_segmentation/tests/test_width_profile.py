import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from measure_width import compute_width_analysis, resolve_measure_outputs  # noqa: E402


def test_compute_width_analysis_orders_single_long_track_and_bins_widths():
    mask = np.zeros((30, 80), dtype=bool)
    mask[10:20, 5:75] = True

    analysis = compute_width_analysis(
        mask,
        pixel_size=2.0,
        bin_size=20.0,
        min_area=1,
    )

    profile = analysis["profile"]
    bins = analysis["bins"]

    assert profile.size > 0
    assert np.all(np.diff(profile["distance_px"]) >= 0)
    assert np.isclose(profile["distance_real"][0], 0.0)
    assert np.allclose(profile["distance_real"], profile["distance_px"] * 2.0)
    assert np.all(profile["width_px"] > 0)

    assert bins.size > 0
    assert np.all(np.diff(bins["start_real"]) >= 0)
    assert np.all(bins["mean_width_real"] > 0)


def test_resolve_measure_outputs_groups_files_by_mask_name(tmp_path):
    outputs = resolve_measure_outputs(
        mask_path=tmp_path / "mask_a.png",
        output_dir=tmp_path / "outputs",
        output_csv=None,
        output_profile_csv=None,
        output_bin_csv=None,
        output_overlay=None,
        output_heatmap=None,
        output_clean_mask=None,
    )

    expected_dir = tmp_path / "outputs" / "mask_a"
    assert outputs["summary_csv"] == expected_dir / "summary.csv"
    assert outputs["profile_csv"] == expected_dir / "width_profile.csv"
    assert outputs["bin_csv"] == expected_dir / "width_bins.csv"
    assert outputs["clean_mask"] == expected_dir / "clean_mask.png"
    assert outputs["overlay"] == expected_dir / "width_overlay.png"
    assert outputs["heatmap"] == expected_dir / "width_heatmap.png"


def test_resolve_measure_outputs_reuses_prediction_group_folder(tmp_path):
    outputs = resolve_measure_outputs(
        mask_path=tmp_path / "outputs" / "sample_001" / "pred_mask.png",
        output_dir=tmp_path / "outputs",
        output_csv=None,
        output_profile_csv=None,
        output_bin_csv=None,
        output_overlay=None,
        output_heatmap=None,
        output_clean_mask=None,
    )

    expected_dir = tmp_path / "outputs" / "sample_001"
    assert outputs["summary_csv"] == expected_dir / "summary.csv"
    assert outputs["heatmap"] == expected_dir / "width_heatmap.png"
