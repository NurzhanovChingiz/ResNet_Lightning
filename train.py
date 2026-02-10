import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from config import CFG
from utils.trainer import Model
import time
from utils.data_preparation import get_dataset, train_test_val_split
from utils.dataset import Dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    try:
        start_time = time.perf_counter()
        
        dataset = get_dataset(CFG.DATA)

        train_images, val_images, test_images, train_labels, val_labels, test_labels = train_test_val_split(
            dataset = dataset,
            test_size = CFG.TEST_SIZE,
            val_size = CFG.VAL_SIZE,
            random_state = CFG.SEED,
            stratify = CFG.STRATIFY
            )
        train_dataset = Dataset(train_images, train_labels, transform=CFG.TRAIN_TRANSFORM)
        val_dataset = Dataset(val_images, val_labels, transform=CFG.TEST_TRANSFORM)
        test_dataset = Dataset(test_images, test_labels, transform=CFG.TEST_TRANSFORM)

        train_dataloader = DataLoader(
            train_dataset, batch_size=CFG.BATCH,
            shuffle=True,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=CFG.PIN_MEMORY,
            persistent_workers=CFG.PERSISTENT_WORKERS,
            worker_init_fn=CFG.SEED_WORKER,
            generator=CFG.GENERATOR,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=CFG.BATCH,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=CFG.PIN_MEMORY,
            persistent_workers=CFG.PERSISTENT_WORKERS,
            worker_init_fn=CFG.SEED_WORKER,
            generator=CFG.GENERATOR,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=CFG.PIN_MEMORY,
            persistent_workers=CFG.PERSISTENT_WORKERS,
            worker_init_fn=CFG.SEED_WORKER,
            generator=CFG.GENERATOR,
        )
        model = Model(model=CFG.MODEL, loss_fn=CFG.LOSS_FN, optimizer=CFG.OPTIMIZER, scheduler=CFG.SCHEDULER, train_transform=CFG.TRAIN_TRANSFORM, test_transform=CFG.TEST_TRANSFORM, print_metrics=CFG.PRINT_METRICS_TO_TERMINAL)
        
        # Setup loggers - TensorBoard and CSV sharing the same auto-incremented version
        tb_logger = TensorBoardLogger(save_dir=".", name="lightning_logs")
        csv_logger = CSVLogger(save_dir=".", name="lightning_logs", version=tb_logger.version)
        
        trainer = L.Trainer(
            accelerator=CFG.ACCELERATOR,
            precision=CFG.PRECISION if CFG.AMP else "32-true",
            max_epochs=CFG.EPOCHS,
            logger=[tb_logger, csv_logger],
            enable_progress_bar=True,
            callbacks=[
                EarlyStopping(
                    min_delta = CFG.MIN_DELTA,
                    monitor="val_loss",
                    patience=CFG.PATIENCE,
                    mode="min"
                ),
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    filename="best-{epoch}-{val_loss:.6f}",
                    dirpath=None,  # auto-saves to the first logger's version folder
                    verbose=True,
                )]
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            )
        # trainer.test(
        #     model=model,
        #     dataloaders=test_dataloader,
        #     ckpt_path='best',
        #     )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Training completed in {elapsed_time/60:.2f} minutes.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        
    finally:
        train_dataloader = None
        val_dataloader = None
        test_dataloader = None
        