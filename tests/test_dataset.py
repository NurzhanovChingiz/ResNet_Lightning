"""Contracts for path-based RGB dataset loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from utils.dataset import Dataset


def test_dataset_loads_rgb_image_and_label(tmp_path: Path) -> None:
    """__getitem__ opens the image as RGB and returns the stored label."""
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image_path)
    dataset = Dataset(data=[str(image_path)], labels=[3])
    sample, label = dataset[0]
    assert label == 3
    assert sample.mode == "RGB"
    assert len(dataset) == 1
