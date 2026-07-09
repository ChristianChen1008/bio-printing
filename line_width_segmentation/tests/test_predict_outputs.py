import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import resolve_predict_outputs  # noqa: E402


def test_resolve_predict_outputs_groups_files_by_image_name(tmp_path):
    outputs = resolve_predict_outputs(
        image_path=tmp_path / "sample_001.png",
        output_dir=tmp_path / "outputs",
        output=None,
    )

    expected_dir = tmp_path / "outputs" / "sample_001"
    assert outputs["mask"] == expected_dir / "pred_mask.png"
    assert outputs["probability"] == expected_dir / "probability.png"
