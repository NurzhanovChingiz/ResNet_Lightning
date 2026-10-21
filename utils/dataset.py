"""Map image paths and labels into a loadable PyTorch dataset."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class Dataset(TorchDataset[tuple[Any, int]]):
    """Load RGB images from paths, apply an optional transform, and reuse tensors.

    Args:
        data: Image file paths.
        labels: Integer class labels aligned with ``data``.
        transform: Optional transform applied after the image is loaded.
        cache_transformed: When True, store transformed tensors in an LRU.
        cache_maxsize: Maximum number of tensors kept when caching is on.
    """

    def __init__(
        self,
        data: Sequence[str],
        labels: Sequence[int],
        transform: Callable[[Any], Any] | None = None,
        *,
        cache_transformed: bool = True,
        cache_maxsize: int = 128,
    ) -> None:
        """Store paths, labels, transform, and a bounded in-process LRU.

        Purpose:
            Configure one dataset instance so recent indices can skip a
            second PIL open, RGB convert, and transform without pinning
            the full split in RAM.

        Strategy:
            Keep constructor values on the instance, then store the cache
            flag, a positive LRU cap, and an empty per-index tensor map.
            Do not look up a shared cache or a service locator.

        Args:
            data: Image file paths.
            labels: Integer class labels aligned with ``data``.
            transform: Optional transform applied after the image is loaded.
            cache_transformed: When True, reuse a transformed tensor by index.
            cache_maxsize: Maximum cached tensors for this instance.

        Returns:
            None.

        Raises:
            ValueError: If caching is on and ``cache_maxsize`` is not positive.

        Flow:
            1. Store production dataset fields
            2. Store cache flag, LRU cap, and empty tensor map
        """
        # 1. Store production dataset fields
        self.data = data
        self.labels = labels
        self.transform = transform
        # 2. Store cache flag, LRU cap, and empty tensor map
        if cache_transformed and cache_maxsize < 1:
            message = "cache_maxsize must be positive when caching is enabled"
            raise ValueError(message)
        self._cache_transformed = cache_transformed
        self._cache_maxsize = cache_maxsize
        self._tensor_cache: OrderedDict[int, Tensor] = OrderedDict()

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Dataset length.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Load one RGB sample, skipping duplicate convert and transform work.

        Purpose:
            Return the transformed tensor and label for ``idx``, without
            repeating PIL ``convert("RGB")`` on an already-RGB image or
            repeating the transform when this index is still in the LRU.

        Strategy:
            Return a clone of a cached tensor on a hit and mark it recent.
            On a miss, open the path, convert only when the mode is not RGB,
            apply the optional transform, and store a tensor in the LRU when
            caching is enabled, evicting the oldest entry when the cap is full.

        Args:
            idx: Sample index.

        Returns:
            Transformed sample and the integer label.

        Flow:
            1. Return cached tensor on a hit
            2. Open image and convert only when not RGB
            3. Apply optional transform
            4. Store a tensor in the LRU and return a clone
        """
        label = self.labels[idx]
        # 1. Return cached tensor on a hit
        if self._cache_transformed:
            cached = self._tensor_cache.get(idx)
            if cached is not None:
                self._tensor_cache.move_to_end(idx)
                return cached.clone(), label
        # 2. Open image and convert only when not RGB
        opened = Image.open(self.data[idx])
        sample: Any = opened if opened.mode == "RGB" else opened.convert("RGB")
        # 3. Apply optional transform
        if self.transform:
            sample = self.transform(sample)
        # 4. Store a tensor in the LRU and return a clone
        if self._cache_transformed and isinstance(sample, Tensor):
            self._tensor_cache[idx] = sample
            self._tensor_cache.move_to_end(idx)
            while len(self._tensor_cache) > self._cache_maxsize:
                self._tensor_cache.popitem(last=False)
            return sample.clone(), label
        return sample, label
