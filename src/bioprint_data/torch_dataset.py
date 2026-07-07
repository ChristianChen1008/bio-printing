from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from bioprint_data.config import DatasetConfig


def default_image_loader(filepath: str):
    from PIL import Image

    with Image.open(filepath) as image:
        return image.convert("RGB")


class FiftyOneBioprintDataset:
    def __init__(
        self,
        samples: Sequence[Any],
        config: DatasetConfig,
        image_loader: Callable[[str], Any] | None = None,
        image_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.samples = samples
        self.config = config
        self.image_loader = image_loader or default_image_loader
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        filepath = sample.filepath
        image = self.image_loader(filepath)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return {
            "image": image,
            "numeric_features": self._numeric_features(sample, filepath),
            "categorical_features": {
                field: sample.get_field(field) for field in self.config.categorical_fields
            },
            "target": self._target(sample),
            "filepath": filepath,
        }

    def _numeric_features(self, sample: Any, filepath: str) -> list[float]:
        features: list[float] = []
        for field in self.config.numeric_fields:
            value = sample.get_field(field)
            features.append(_coerce_numeric_feature(field, filepath, value))
        return features

    def _target(self, sample: Any) -> Any:
        if self.config.target_column:
            return sample.get_field(self.config.target_column)
        return None


def _coerce_numeric_feature(field: str, filepath: str, value: Any) -> float:
    if value is None:
        raise ValueError(
            f"Invalid numeric field '{field}' for sample filepath '{filepath}': {value!r}"
        )

    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric field '{field}' for sample filepath '{filepath}': {value!r}"
        ) from exc

    if not math.isfinite(converted):
        raise ValueError(
            f"Invalid numeric field '{field}' for sample filepath '{filepath}': {value!r}"
        )

    return converted
