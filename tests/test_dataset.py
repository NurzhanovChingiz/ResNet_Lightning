"""Contracts for path-based RGB dataset loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2

from utils.dataset import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _write_rgb_png(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    """Write one RGB PNG used by dataset contract tests.

    Args:
        path: Destination image path.
        color: Solid fill color.

    Returns:
        None.
    """
    Image.new("RGB", (2, 2), color=color).save(path)


def _to_tensor() -> v2.Compose:
    """Return a deterministic path-to-tensor transform.

    Returns:
        Compose that converts a PIL image into a scaled float tensor.
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ]
    )


def _counting_transform() -> tuple[Callable[[Image.Image], Tensor], dict[str, int]]:
    """Return a tensor transform that counts how many times it runs.

    Returns:
        Transform and a mutable call-count mapping.
    """
    calls = {"count": 0}
    inner = _to_tensor()

    def counting_transform(image: Image.Image) -> Tensor:
        calls["count"] += 1
        converted = inner(image)
        if not isinstance(converted, Tensor):
            message = "transform must return a tensor"
            raise TypeError(message)
        return converted

    return counting_transform, calls


def test_dataset_loads_rgb_image_and_label(tmp_path: Path) -> None:
    """__getitem__ opens the image as RGB and returns the stored label."""
    image_path = tmp_path / "sample.png"
    _write_rgb_png(image_path)
    dataset = Dataset(data=[str(image_path)], labels=[3])
    sample, label = dataset[0]
    assert label == 3
    assert sample.mode == "RGB"
    assert len(dataset) == 1


def test_repeated_index_skips_second_transform(tmp_path: Path) -> None:
    """A cached tensor hit must not run the transform a second time."""
    image_path = tmp_path / "sample.png"
    _write_rgb_png(image_path)
    counting_transform, calls = _counting_transform()
    dataset = Dataset(
        data=[str(image_path)],
        labels=[3],
        transform=counting_transform,
        cache_transformed=True,
        cache_maxsize=1,
    )
    first, first_label = dataset[0]
    first_snapshot = first.clone()
    first.fill_(0)
    second, second_label = dataset[0]
    assert first_label == 3
    assert second_label == 3
    assert calls["count"] == 1
    assert torch.equal(second, first_snapshot)


def test_disabled_cache_reruns_transform(tmp_path: Path) -> None:
    """cache_transformed=False must reload and transform on every access."""
    image_path = tmp_path / "sample.png"
    _write_rgb_png(image_path)
    counting_transform, calls = _counting_transform()
    dataset = Dataset(
        data=[str(image_path)],
        labels=[4],
        transform=counting_transform,
        cache_transformed=False,
        cache_maxsize=1,
    )
    dataset[0]
    dataset[0]
    assert calls["count"] == 2


def test_already_rgb_image_skips_convert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An already-RGB PNG must not call PIL convert on the load path."""
    image_path = tmp_path / "sample.png"
    _write_rgb_png(image_path)
    calls = {"count": 0}
    original_convert = Image.Image.convert

    def counting_convert(self: Image.Image, mode: str) -> Image.Image:
        calls["count"] += 1
        return original_convert(self, mode)

    monkeypatch.setattr(Image.Image, "convert", counting_convert)
    dataset = Dataset(
        data=[str(image_path)],
        labels=[0],
        cache_transformed=False,
    )
    sample, label = dataset[0]
    assert label == 0
    assert sample.mode == "RGB"
    assert calls["count"] == 0


def test_dataset_instances_keep_private_caches(tmp_path: Path) -> None:
    """Mutating one instance's returned tensor must not change a second instance."""
    image_path = tmp_path / "sample.png"
    _write_rgb_png(image_path, color=(255, 0, 0))
    transform = _to_tensor()
    first_dataset = Dataset(
        data=[str(image_path)],
        labels=[1],
        transform=transform,
        cache_transformed=True,
    )
    second_dataset = Dataset(
        data=[str(image_path)],
        labels=[1],
        transform=transform,
        cache_transformed=True,
    )
    first, _ = first_dataset[0]
    expected, _ = second_dataset[0]
    first.fill_(0)
    reread, _ = second_dataset[0]
    assert torch.equal(reread, expected)
    assert not torch.equal(reread, first)


def test_lru_evicts_oldest_index(tmp_path: Path) -> None:
    """A full LRU must drop the oldest index and rerun that transform."""
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    _write_rgb_png(first_path, color=(10, 20, 30))
    _write_rgb_png(second_path, color=(40, 50, 60))
    counting_transform, calls = _counting_transform()
    dataset = Dataset(
        data=[str(first_path), str(second_path)],
        labels=[1, 2],
        transform=counting_transform,
        cache_transformed=True,
        cache_maxsize=1,
    )
    dataset[0]
    dataset[0]
    dataset[1]
    dataset[0]
    assert calls["count"] == 3


def test_cache_maxsize_rejects_zero_when_caching() -> None:
    """Caching with a non-positive LRU cap must fail closed."""
    with pytest.raises(ValueError, match="cache_maxsize must be positive"):
        Dataset(
            data=["unused.png"], labels=[0], cache_transformed=True, cache_maxsize=0
        )
