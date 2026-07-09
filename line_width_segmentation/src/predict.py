import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def pad_to_patch(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return padded, pad_h, pad_w


def predict_sliding_window(
    model: Any,
    image_rgb: np.ndarray,
    patch_size: int,
    overlap: int,
    device: Any,
) -> np.ndarray:
    import torch

    if overlap >= patch_size:
        raise ValueError("--overlap must be smaller than --patch-size")

    padded, pad_h, pad_w = pad_to_patch(image_rgb, patch_size)
    height, width = padded.shape[:2]
    stride = patch_size - overlap

    prob_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    y_positions = list(range(0, max(height - patch_size, 0) + 1, stride))
    x_positions = list(range(0, max(width - patch_size, 0) + 1, stride))
    if y_positions[-1] != height - patch_size:
        y_positions.append(height - patch_size)
    if x_positions[-1] != width - patch_size:
        x_positions.append(width - patch_size)

    model.eval()
    with torch.no_grad():
        for y in y_positions:
            for x in x_positions:
                patch = padded[y:y + patch_size, x:x + patch_size]
                tensor = patch.astype(np.float32) / 255.0
                tensor = np.transpose(tensor, (2, 0, 1))
                tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)

                logits = model(tensor)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

                prob_sum[y:y + patch_size, x:x + patch_size] += prob
                count[y:y + patch_size, x:x + patch_size] += 1.0

    probability = prob_sum / np.maximum(count, 1e-7)
    if pad_h:
        probability = probability[:-pad_h, :]
    if pad_w:
        probability = probability[:, :-pad_w]
    return probability


def resolve_predict_outputs(
    image_path: str | Path,
    output_dir: str | Path | None,
    output: str | Path | None,
) -> dict[str, Path]:
    if output_dir is None:
        if output is None:
            raise ValueError("Either --output or --output-dir is required.")
        mask_path = Path(output)
        return {
            "mask": mask_path,
            "probability": mask_path.with_name(mask_path.stem + "_probability.png"),
        }

    group_dir = Path(output_dir) / Path(image_path).stem
    return {
        "mask": Path(output) if output else group_dir / "pred_mask.png",
        "probability": group_dir / "probability.png",
    }


def main() -> None:
    import segmentation_models_pytorch as smp
    import torch

    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")

    encoder = args.encoder or checkpoint.get("encoder", "efficientnet-b1")
    patch_size = args.patch_size or checkpoint.get("patch_size", 768)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    probability = predict_sliding_window(
        model=model,
        image_rgb=image_rgb,
        patch_size=patch_size,
        overlap=args.overlap,
        device=device,
    )
    mask = (probability > args.threshold).astype(np.uint8) * 255

    outputs = resolve_predict_outputs(args.image, args.output_dir, args.output)
    output_path = outputs["mask"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)

    probability_path = outputs["probability"]
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(probability_path), (probability * 255).astype(np.uint8))
    print(f"Saved mask: {output_path}")
    print(f"Saved probability map: {probability_path}")


if __name__ == "__main__":
    main()
