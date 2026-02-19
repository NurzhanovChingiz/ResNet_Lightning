import lightning as L
import torch
import torch.nn as nn
from typing import Any
from utils.metrics import MetricsTracker


class Model(L.LightningModule):
    '''
    LightningModule for training a model.
    Args:
        model: nn.Module
        loss_fn: nn.Module
        optimizer: torch.optim.Optimizer
        train_transform: AnyS
        test_transform: Any
    Returns:
        LightningModule
    '''
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, scheduler: dict | None = None, train_transform: Any = None, test_transform: Any = None, print_metrics: bool = False):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.print_metrics = print_metrics
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'optimizer', 'scheduler', 'train_transform', 'test_transform', 'print_metrics'])
        
        # Metrics trackers
        self.train_metrics_tracker = MetricsTracker()
        self.val_metrics_tracker = MetricsTracker()
        self.test_metrics_tracker = MetricsTracker()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Any:
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics
        self.train_metrics_tracker.update_metrics(y_hat, y)
        self.train_metrics_tracker.update_loss(loss.item())
        
        # Log step metrics (shown in progress bar)
        self.log('train_loss_step', self.train_metrics_tracker.get_metric('avg_loss'), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        # Compute and log epoch metrics
        train_metrics = self.train_metrics_tracker.compute()
        self.log('train_loss', self.train_metrics_tracker.get_metric('avg_loss').compute().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_top1', self.train_metrics_tracker.get_metric('accuracy_top1').compute().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_top5', self.train_metrics_tracker.get_metric('accuracy_top5').compute().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_precision', self.train_metrics_tracker.get_metric('precision').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_recall', self.train_metrics_tracker.get_metric('recall').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_f1', self.train_metrics_tracker.get_metric('f1_score').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        if self.print_metrics:
            print("Training Metrics:", end=" | ")
            self.train_metrics_tracker.print_metrics()
            print(flush=True)
        # Reset metrics
        self.train_metrics_tracker.reset()
    
    def on_validation_epoch_end(self) -> None:
        # Compute and log epoch metrics
        val_metrics = self.val_metrics_tracker.compute()
        
        self.log('val_loss', self.val_metrics_tracker.get_metric('avg_loss').compute().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_top1', self.val_metrics_tracker.get_metric('accuracy_top1').compute().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_top5', self.val_metrics_tracker.get_metric('accuracy_top5').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_precision', self.val_metrics_tracker.get_metric('precision').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', self.val_metrics_tracker.get_metric('recall').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', self.val_metrics_tracker.get_metric('f1_score').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        if self.print_metrics:
            print("Validation Metrics:", end=" | ")
            self.val_metrics_tracker.print_metrics()
            print(flush=True)
        # Reset metrics
        self.val_metrics_tracker.reset()
    
    def on_test_epoch_end(self) -> None:
        # Compute and log epoch metrics
        test_metrics = self.test_metrics_tracker.compute()

        self.log('test_loss', self.test_metrics_tracker.get_metric('avg_loss').compute().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_top1', self.test_metrics_tracker.get_metric('accuracy_top1').compute().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_top5', self.test_metrics_tracker.get_metric('accuracy_top5').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_precision', self.test_metrics_tracker.get_metric('precision').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_recall', self.test_metrics_tracker.get_metric('recall').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_f1', self.test_metrics_tracker.get_metric('f1_score').compute().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        if self.print_metrics:
            print("Test Metrics:", end=" | ")
            self.test_metrics_tracker.print_metrics()
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
    