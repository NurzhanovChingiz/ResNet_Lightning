import torch
from torchmetrics import (
    MeanMetric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    
)

from config import CFG
from torchmetrics.aggregation import BaseAggregator

def suppress_nan_check(MetricClass):
    assert issubclass(MetricClass, BaseAggregator), MetricClass
    class DisableNanCheck(MetricClass):
        def _cast_and_nan_check_input(self, x, weight=None):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            x = x.to(device=self.device, dtype=self.dtype)
            if weight is not None and not isinstance(weight, torch.Tensor):
                weight = torch.as_tensor(weight)
            if weight is None:
                weight = torch.ones_like(x)
            weight = weight.to(device=self.device, dtype=self.dtype)
            return x, weight
    return DisableNanCheck

class MetricsTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        NoNanMeanMetric = suppress_nan_check(MeanMetric)
        self.metrics = torch.nn.ModuleDict({
            "avg_loss": NoNanMeanMetric(),
            "accuracy_top1": Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=1),
            "accuracy_top5": Accuracy(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='weighted', top_k=5),
            "precision": Precision(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro'),
            "recall": Recall(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro'),
            "f1_score": F1Score(task=CFG.TASK, num_classes=CFG.NUM_CLASSES, average='macro'),
        })
    
    def update_metrics(self, preds, targets):
        self.metrics["accuracy_top1"].update(preds, targets)
        self.metrics["accuracy_top5"].update(preds, targets)
        self.metrics["precision"].update(preds, targets)
        self.metrics["recall"].update(preds, targets)
        self.metrics["f1_score"].update(preds, targets)
        
    def update_loss(self, loss_value):
        loss_value = torch.as_tensor(loss_value, dtype=self.metrics["avg_loss"].dtype, device=CFG.DEVICE)
        self.metrics["avg_loss"].update(loss_value, weight=torch.ones_like(loss_value))
        
    def compute(self):
        return {name: metric.compute().item() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
            
        
    def get_metric(self, name: str):
        return self.metrics[name]
    
    def print_metrics(self):
        # Check if any updates have been made before computing
        # MeanMetric uses mean_value as state, classification metrics use preds/target
        avg_loss_metric = self.metrics["avg_loss"]
        if avg_loss_metric.weight.sum() == 0:
            # No updates have been made, skip printing
            return 
        computed_metrics = self.compute()
        for name, value in computed_metrics.items():
            # in one line
            print(f"{name}: {value:.6f}", end=' | ')
        # Ensure the line is finalized; otherwise tqdm may overwrite it.
        print(flush=True)
        self.reset()