"""Contracts for ImageFolder loading and stratified splits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from utils.data_preparation import get_dataset, train_test_val_split


def _write_image_folder(root: Path, *, images_per_class: int) -> None:
    for class_name in ("a", "b"):
        class_dir = root / class_name
        class_dir.mkdir()
        for index in range(images_per_class):
            Image.new("RGB", (2, 2)).save(class_dir / f"{index}.png")


def test_get_dataset_loads_image_folder(tmp_path: Path) -> None:
    """get_dataset returns an ImageFolder bound to the given root."""
    _write_image_folder(tmp_path, images_per_class=1)
    dataset = get_dataset(str(tmp_path))
    assert len(dataset.imgs) == 2


def test_train_test_val_split_preserves_counts(tmp_path: Path) -> None:
    """A stratified split returns disjoint path lists whose sizes add up."""
    _write_image_folder(tmp_path, images_per_class=10)
    dataset = get_dataset(str(tmp_path))
    train_images, val_images, test_images, train_labels, val_labels, test_labels = (
        train_test_val_split(
            dataset,
            test_size=0.2,
            val_size=0.25,
            random_state=0,
            stratify=True,
        )
    )
    assert len(train_images) == len(train_labels)
    assert len(val_images) == len(val_labels)
    assert len(test_images) == len(test_labels)
    assert len(train_images) + len(val_images) + len(test_images) == 20
    assert not set(train_images) & set(val_images)
    assert not set(train_images) & set(test_images)
    assert not set(val_images) & set(test_images)
