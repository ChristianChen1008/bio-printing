import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrackDataset, build_train_transform, build_val_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-masks", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-masks", required=True)
    parser.add_argument("--output", default="outputs/best_model.pth")
    parser.add_argument("--encoder", default="efficientnet-b1")
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def dice_score_from_logits(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-7) / (union + 1e-7)).mean()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    dice_loss: torch.nn.Module,
    bce_loss: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    threshold: float,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = dice_loss(logits, masks) + bce_loss(logits, masks)
            dice = dice_score_from_logits(logits, masks, threshold)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_dice += dice.item()

    return total_loss / len(loader), total_dice / len(loader)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TrackDataset(
        args.train_images,
        args.train_masks,
        transform=build_train_transform(args.patch_size),
    )
    val_dataset = TrackDataset(
        args.val_images,
        args.val_masks,
        transform=build_val_transform(args.patch_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(
            model,
            train_loader,
            dice_loss,
            bce_loss,
            device,
            optimizer,
            args.threshold,
        )
        val_loss, val_dice = run_epoch(
            model,
            val_loader,
            dice_loss,
            bce_loss,
            device,
            optimizer=None,
            threshold=args.threshold,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f}, dice {train_dice:.4f} | "
            f"val loss {val_loss:.4f}, dice {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder": args.encoder,
                    "patch_size": args.patch_size,
                    "best_dice": best_dice,
                },
                output_path,
            )
            print(f"Saved best model to {output_path} with dice {best_dice:.4f}")


if __name__ == "__main__":
    main()
