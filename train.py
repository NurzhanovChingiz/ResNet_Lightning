"""Train the configured Lightning model and log metrics to CSV."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from config import CFG
from utils.data_preparation import get_dataset, train_test_val_split
from utils.dataset import Dataset
from utils.trainer import Model

if TYPE_CHECKING:
    from typing import Any

LOGGER = logging.getLogger(__name__)


def build_split_datasets() -> tuple[Dataset, Dataset, Dataset]:
    """Build train, validation, and test datasets from CFG.

    Purpose:
        Load the image folder, split paths and labels, and wrap each split.

    Strategy:
        Use CFG split sizes, seed, and stratify flag, then apply the train
        transform to the training split and the test transform to the others.

    Returns:
        Train, validation, and test ``Dataset`` instances.

    Flow:
        1. Load the image folder
        2. Split paths and labels
        3. Wrap each split in Dataset
    """
    # 1. Load the image folder
    dataset = get_dataset(CFG.DATA)

    # 2. Split paths and labels
    train_images, val_images, test_images, train_labels, val_labels, test_labels = (
        train_test_val_split(
            dataset=dataset,
            test_size=CFG.TEST_SIZE,
            val_size=CFG.VAL_SIZE,
            random_state=CFG.SEED,
            stratify=CFG.STRATIFY,
        )
    )

    # 3. Wrap each split in Dataset
    train_dataset = Dataset(train_images, train_labels, transform=CFG.TRAIN_TRANSFORM)
    val_dataset = Dataset(val_images, val_labels, transform=CFG.TEST_TRANSFORM)
    test_dataset = Dataset(test_images, test_labels, transform=CFG.TEST_TRANSFORM)
    return train_dataset, val_dataset, test_dataset


def build_dataloader(
    dataset: Dataset,
    *,
    shuffle: bool,
    batch_size: int,
) -> DataLoader[tuple[Any, int]]:
    """Wrap ``dataset`` in a DataLoader using CFG worker settings.

    Args:
        dataset: Split dataset to iterate.
        shuffle: Whether to shuffle samples each epoch.
        batch_size: Mini-batch size.

    Returns:
        DataLoader over ``dataset``.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=CFG.PIN_MEMORY,
        persistent_workers=CFG.PERSISTENT_WORKERS,
    )


def build_lightning_model() -> Model:
    """Construct the Lightning module from CFG training settings.

    Returns:
        Configured ``Model`` instance.
    """
    return Model(
        model=CFG.MODEL,
        loss_fn=CFG.LOSS_FN,
        optimizer=CFG.OPTIMIZER,
        scheduler=CFG.SCHEDULER,
        task=CFG.TASK,
        num_classes=CFG.NUM_CLASSES,
        device=CFG.DEVICE,
        train_transform=CFG.TRAIN_TRANSFORM,
        test_transform=CFG.TEST_TRANSFORM,
        print_metrics=CFG.PRINT_METRICS_TO_TERMINAL,
    )


def build_trainer() -> Trainer:
    """Construct the Lightning trainer with CSV logging and callbacks.

    Purpose:
        Assemble the trainer that fits the model using CFG hardware and
        stopping settings.

    Strategy:
        Log only with CSVLogger, then attach early stopping and a single
        best-checkpoint callback before constructing the Trainer.

    Returns:
        Configured Lightning Trainer.

    Flow:
        1. Create CSV logger
        2. Construct Trainer with callbacks
    """
    # 1. Create CSV logger
    csv_logger = CSVLogger(save_dir=".", name="lightning_logs", version="")
    # 2. Construct Trainer with callbacks
    return Trainer(
        accelerator=CFG.ACCELERATOR,
        precision=CFG.PRECISION if CFG.AMP else "32-true",
        max_epochs=CFG.EPOCHS,
        logger=csv_logger,
        enable_progress_bar=True,
        callbacks=[
            EarlyStopping(
                min_delta=CFG.MIN_DELTA,
                monitor="val_loss",
                patience=CFG.PATIENCE,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch}-{val_loss:.6f}",
                dirpath="lightning_logs/checkpoints/",
                verbose=True,
            ),
        ],
    )


def run_training() -> tuple[
    DataLoader[tuple[Any, int]],
    DataLoader[tuple[Any, int]],
    DataLoader[tuple[Any, int]],
]:
    """Run dataset, loader, model, and trainer stages and fit the model.

    Purpose:
        Execute the training workflow from CFG-backed datasets through
        ``trainer.fit``, then log elapsed minutes.

    Strategy:
        Build each training dependency as a peer stage, fit the model, and
        return the three dataloaders so the CLI entry can release them.

    Returns:
        Train, validation, and test dataloaders used during the run.

    Flow:
        1. Start timer
        2. Build split datasets
        3. Build dataloaders
        4. Build Lightning model
        5. Build trainer
        6. Fit the model
        7. Report elapsed minutes
    """
    # 1. Start timer
    start_time = time.perf_counter()

    # 2. Build split datasets
    train_dataset, val_dataset, test_dataset = build_split_datasets()

    # 3. Build dataloaders
    train_dataloader = build_dataloader(
        train_dataset, shuffle=True, batch_size=CFG.BATCH
    )
    val_dataloader = build_dataloader(val_dataset, shuffle=False, batch_size=CFG.BATCH)
    test_dataloader = build_dataloader(test_dataset, shuffle=False, batch_size=1)

    # 4. Build Lightning model
    model = build_lightning_model()

    # 5. Build trainer
    trainer = build_trainer()

    # 6. Fit the model
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 7. Report elapsed minutes
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    LOGGER.info("Training completed in %.2f minutes.", elapsed_time / 60)
    return train_dataloader, val_dataloader, test_dataloader


def main() -> None:
    """Invoke training and keep dataloader cleanup here.

    Purpose:
        Run the configured training workflow and keep dataloader release
        in this CLI entry.

    Strategy:
        Call ``run_training`` and drop the dataloader references in
        ``finally`` so a training exception still releases loaders.

    Returns:
        None. Elapsed time is logged from ``run_training``. Training
        exceptions propagate after cleanup.

    Flow:
        1. Run training
        2. Release dataloaders
    """
    train_dataloader = None
    val_dataloader = None
    test_dataloader = None
    try:
        # 1. Run training
        train_dataloader, val_dataloader, test_dataloader = run_training()
    finally:
        # 2. Release dataloaders
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        _released = (train_dataloader, val_dataloader, test_dataloader)
        del _released


if __name__ == "__main__":
    main()
