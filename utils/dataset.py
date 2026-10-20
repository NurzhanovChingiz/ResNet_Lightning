"""Map image paths and labels into a loadable PyTorch dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PIL import Image
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class Dataset(TorchDataset[tuple[Any, int]]):
    """Load RGB images from paths and apply an optional transform.

    Args:
        data: Image file paths.
        labels: Integer class labels aligned with ``data``.
        transform: Optional transform applied after the image is loaded.
    """

    def __init__(
        self,
        data: Sequence[str],
        labels: Sequence[int],
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        """Store paths, labels, and the optional image transform.

        Args:
            data: Image file paths.
            labels: Integer class labels aligned with ``data``.
            transform: Optional transform applied after the image is loaded.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Dataset length.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Load one RGB image and its label.

        Strategy:
            Open the indexed path as RGB, read the aligned label, apply the
            optional transform when set, then return the pair.

        Args:
            idx: Sample index.

        Returns:
            RGB image, transformed only when transform is set, plus the integer
            label.

        Flow:
            1. Load RGB image
            2. Read aligned label
            3. Apply optional transform
            4. Return sample and label
        """
        # 1. Load RGB image
        sample = Image.open(self.data[idx]).convert("RGB")
        # 2. Read aligned label
        label = self.labels[idx]
        # 3. Apply optional transform
        if self.transform:
            sample = self.transform(sample)
        # 4. Return sample and label
        return sample, label
