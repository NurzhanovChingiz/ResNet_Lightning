"""Load image folders and split them into train, validation, and test sets."""

from __future__ import annotations

import logging

from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

LOGGER = logging.getLogger(__name__)


def get_dataset(folder: str) -> ImageFolder:
    """Load an ImageFolder dataset from a class-subdirectory root.

    Args:
        folder: Path to the dataset folder.

    Returns:
        The torchvision ImageFolder bound to ``folder``.
    """
    return ImageFolder(root=folder)


def train_test_val_split(
    dataset: ImageFolder,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    *,
    stratify: bool = True,
) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
    """Split image paths and labels into train, validation, and test sets.

    Strategy:
        Hold out a test split from the full ImageFolder, then split the
        remainder into train and validation. Stratify both cuts when requested
        so class proportions stay aligned.

    Args:
        dataset: The dataset to split.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the remaining data to include in validation.
        random_state: Random seed for reproducibility.
        stratify: If True, preserve class proportions in each split.

    Returns:
        Train, validation, and test image paths followed by their labels.

    Flow:
        1. Extract image paths and labels
        2. Hold out the test split
        3. Split remaining data into train and validation
        4. Report split sizes
        5. Return the six lists
    """
    # 1. Extract image paths and labels
    images_list = [path for path, _ in dataset.imgs]
    label_list = [label for _, label in dataset.imgs]
    # 2. Hold out the test split
    train_images, test_images, train_labels, test_labels = train_test_split(
        images_list,
        label_list,
        test_size=test_size,
        random_state=random_state,
        stratify=label_list if stratify else None,
    )
    # 3. Split remaining data into train and validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images,
        train_labels,
        test_size=val_size,
        random_state=random_state,
        stratify=train_labels if stratify else None,
    )
    # 4. Report split sizes
    train_percent = len(train_images) / len(images_list) * 100
    val_percent = len(val_images) / len(images_list) * 100
    test_percent = len(test_images) / len(images_list) * 100
    LOGGER.info(
        "After split - Train: %s, Val: %s, Test: %s",
        len(train_images),
        len(val_images),
        len(test_images),
    )
    LOGGER.info(
        "After split percentages - Train: %.2f%%, Val: %.2f%%, Test: %.2f%%",
        train_percent,
        val_percent,
        test_percent,
    )
    # 5. Return the six lists
    return train_images, val_images, test_images, train_labels, val_labels, test_labels
