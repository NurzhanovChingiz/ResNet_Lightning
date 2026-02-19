import lightning as L
import torch
import torch.nn as nn
from typing import Any, Literal
from utils.metrics import MetricsTracker


class Model(L.LightningModule):
    '''
    LightningModule for training a model.
    Args:
        model: nn.Module
        loss_fn: nn.Module
        optimizer: torch.optim.Optimizer
        train_transform: Any
        test_transform: Any
        print_metrics: bool
        task: Literal["binary", "multiclass", "multilabel"]
        num_classes: int
    Returns:
        LightningModule
    '''
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: dict | None = None,
        train_transform: Any = None,
        test_transform: Any = None,
        print_metrics: bool = False,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        num_classes: int = 1000,
        ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.print_metrics = print_metrics
        self.task = task
        self.num_classes = num_classes
        
        # Metrics trackers
        self.train_metrics_tracker = MetricsTracker(task=task, num_classes=num_classes)
        self.val_metrics_tracker = MetricsTracker(task=task, num_classes=num_classes)
        self.test_metrics_tracker = MetricsTracker(task=task, num_classes=num_classes)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Any:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics
        self.train_metrics_tracker.update_metrics(y_hat, y)
        self.train_metrics_tracker.update_loss(loss.item())
        
        # Log step metrics (shown in progress bar)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        # Compute and log epoch metrics
        metrics = self.train_metrics_tracker.compute()
        self.log('train_loss', metrics['avg_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_top1', metrics['accuracy_top1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_top5', metrics['accuracy_top5'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_precision', metrics['precision'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_recall', metrics['recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_f1', metrics['f1_score'], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if self.print_metrics:
            print("Training Metrics:", end=" | ")
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}", end=' | ')
            print(flush=True)
        # Reset metrics
        self.train_metrics_tracker.reset()
    
    def on_validation_epoch_end(self) -> None:
        # Compute and log epoch metrics
        metrics = self.val_metrics_tracker.compute()
        self.log('val_loss', metrics['avg_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_top1', metrics['accuracy_top1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_top5', metrics['accuracy_top5'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_precision', metrics['precision'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', metrics['recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', metrics['f1_score'], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if self.print_metrics:
            print("Validation Metrics:", end=" | ")
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}", end=' | ')
            print(flush=True)
        # Reset metrics
        self.val_metrics_tracker.reset()
    
    def on_test_epoch_end(self) -> None:
        # Compute and log epoch metrics
        metrics = self.test_metrics_tracker.compute()
        self.log('test_loss', metrics['avg_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_top1', metrics['accuracy_top1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_top5', metrics['accuracy_top5'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_precision', metrics['precision'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_recall', metrics['recall'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_f1', metrics['f1_score'], on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        if self.print_metrics:
            print("Test Metrics:", end=" | ")
            for name, value in metrics.items():
                print(f"{name}: {value:.6f}", end=' | ')
            print(flush=True)
        # Reset metrics
        self.test_metrics_tracker.reset()
    
    def configure_optimizers(self) -> Any:
        if self.scheduler is not None:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
        return self.optimizer
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Any:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics
        self.val_metrics_tracker.update_metrics(y_hat, y)
        self.val_metrics_tracker.update_loss(loss.item())
        
        return loss
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Any:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics
        self.test_metrics_tracker.update_metrics(y_hat, y)
        self.test_metrics_tracker.update_loss(loss.item())
        return loss
    