import lightning as L
import torch
import torch.nn as nn
from typing import Any
from torchmetrics import (
    MeanMetric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    
)
from config import CFG


class Model(L.LightningModule):
    '''
    LightningModule for training a model.
    Args:
        model: nn.Module
        loss_fn: nn.Module
        optimizer: torch.optim.Optimizer
        train_transform: Any
        test_transform: Any
    Returns:
        LightningModule
    '''
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, train_transform: Any, test_transform: Any):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'optimizer', 'train_transform', 'test_transform'])
        
        # Training metrics
        self.train_loss = MeanMetric()
        self.train_accuracy_top1 = Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=1)
        self.train_accuracy_top5 = Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=5)
        self.train_precision = Precision(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        self.train_recall = Recall(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        self.train_f1_score = F1Score(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        
        # Validation metrics
        self.val_loss = MeanMetric()
        self.val_accuracy_top1 = Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=1)
        self.val_accuracy_top5 = Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=5)
        self.val_precision = Precision(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        self.val_recall = Recall(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        self.val_f1_score = F1Score(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        
        # Test metrics
        self.test_loss = MeanMetric()
        self.test_accuracy_top1 = Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=1)
        self.test_accuracy_top5 = Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=5)
        self.test_precision = Precision(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        self.test_recall = Recall(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
        self.test_f1_score = F1Score(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro')
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics
        self.train_loss.update(loss.item())
        self.train_accuracy_top1.update(y_hat, y)
        self.train_accuracy_top5.update(y_hat, y)
        self.train_precision.update(y_hat, y)
        self.train_recall.update(y_hat, y)
        self.train_f1_score.update(y_hat, y)
        
        # Log step metrics (shown in progress bar)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        # Compute and log epoch metrics
        train_loss = self.train_loss.compute()
        train_acc1 = self.train_accuracy_top1.compute()
        train_acc5 = self.train_accuracy_top5.compute()
        train_prec = self.train_precision.compute()
        train_rec = self.train_recall.compute()
        train_f1 = self.train_f1_score.compute()
        
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_top1', train_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_top5', train_acc5, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_precision', train_prec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_recall', train_rec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_f1', train_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # # Print metrics to terminal
        # print(f"\n[Train Epoch {self.current_epoch}] "
        #       f"Loss: {train_loss:.4f} | "
        #       f"Acc@1: {train_acc1:.4f} | "
        #       f"Acc@5: {train_acc5:.4f} | "
        #       f"Precision: {train_prec:.4f} | "
        #       f"Recall: {train_rec:.4f} | "
        #       f"F1: {train_f1:.4f}")
        
        # Reset metrics
        self.train_loss.reset()
        self.train_accuracy_top1.reset()
        self.train_accuracy_top5.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1_score.reset()
    
    def on_validation_epoch_end(self):
        # Compute and log epoch metrics
        val_loss = self.val_loss.compute()
        val_acc1 = self.val_accuracy_top1.compute()
        val_acc5 = self.val_accuracy_top5.compute()
        val_prec = self.val_precision.compute()
        val_rec = self.val_recall.compute()
        val_f1 = self.val_f1_score.compute()
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_top1', val_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc_top5', val_acc5, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_precision', val_prec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', val_rec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', val_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        

        # Reset metrics
        self.val_loss.reset()
        self.val_accuracy_top1.reset()
        self.val_accuracy_top5.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1_score.reset()
    
    def on_test_epoch_end(self):
        # Compute and log epoch metrics
        test_loss = self.test_loss.compute()
        test_acc1 = self.test_accuracy_top1.compute()
        test_acc5 = self.test_accuracy_top5.compute()
        test_prec = self.test_precision.compute()
        test_rec = self.test_recall.compute()
        test_f1 = self.test_f1_score.compute()
        
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_top1', test_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc_top5', test_acc5, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_precision', test_prec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_recall', test_rec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_f1', test_f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        # Print metrics to terminal
        # print(f"\n[Test] "
        #       f"Loss: {test_loss:.4f} | "
        #       f"Acc@1: {test_acc1:.4f} | "
        #       f"Acc@5: {test_acc5:.4f} | "
        #       f"Precision: {test_prec:.4f} | "
        #       f"Recall: {test_rec:.4f} | "
        #       f"F1: {test_f1:.4f}")
        
        # Reset metrics
        self.test_loss.reset()
        self.test_accuracy_top1.reset()
        self.test_accuracy_top5.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1_score.reset()
    
    def configure_optimizers(self):
        return self.optimizer
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics
        self.val_loss.update(loss.item())
        self.val_accuracy_top1.update(y_hat, y)
        self.val_accuracy_top5.update(y_hat, y)
        self.val_precision.update(y_hat, y)
        self.val_recall.update(y_hat, y)
        self.val_f1_score.update(y_hat, y)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        
        # Update metrics
        self.test_loss.update(loss.item())
        self.test_accuracy_top1.update(y_hat, y)
        self.test_accuracy_top5.update(y_hat, y)
        self.test_precision.update(y_hat, y)
        self.test_recall.update(y_hat, y)
        self.test_f1_score.update(y_hat, y)
        
        return loss
    