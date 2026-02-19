from typing import cast, Literal

import torch
from torchmetrics import (
    MeanMetric,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    Metric,
)

from torchmetrics.aggregation import BaseAggregator


def suppress_nan_check(MetricClass: type[BaseAggregator]) -> type:
    assert issubclass(MetricClass, BaseAggregator), MetricClass
    class DisableNanCheck(MetricClass):  # type: ignore[valid-type, misc]
        def _cast_and_nan_check_input(self, x: torch.Tensor | float, weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            x = x.to(device=self.device, dtype=torch.float32)
            if weight is not None and not isinstance(weight, torch.Tensor):
                weight = torch.as_tensor(weight)
            if weight is None:
                weight = torch.ones_like(x)
            weight = weight.to(device=self.device, dtype=torch.float32)
            return x, weight
    return DisableNanCheck

class MetricsTracker(torch.nn.Module):
    def __init__(self, task: Literal["binary", "multiclass", "multilabel"] = "multiclass", num_classes: int = 1000) -> None:
        super().__init__()
        NoNanMeanMetric = suppress_nan_check(MeanMetric)
        self.task = task
        self.num_classes = num_classes
        self.metrics = torch.nn.ModuleDict({
            "avg_loss": NoNanMeanMetric(),
            "accuracy_top1": Accuracy(task=self.task, num_classes=self.num_classes, average='weighted', top_k=1),
            "accuracy_top5": Accuracy(task=self.task, num_classes=self.num_classes, average='weighted', top_k=5),
            "precision": Precision(task=self.task, num_classes=self.num_classes, average='macro'),
            "recall": Recall(task=self.task, num_classes=self.num_classes, average='macro'),
            "f1_score": F1Score(task=self.task, num_classes=self.num_classes, average='macro'),
        })
    
    def _metric(self, name: str) -> Metric:
        return cast(Metric, self.metrics[name])

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self._metric("accuracy_top1").update(preds, targets)
        self._metric("accuracy_top5").update(preds, targets)
        self._metric("precision").update(preds, targets)
        self._metric("recall").update(preds, targets)
        self._metric("f1_score").update(preds, targets)

    def update_loss(self, loss_value: float) -> None:
        avg_loss = cast(MeanMetric, self.metrics["avg_loss"])
        loss_tensor = torch.as_tensor(loss_value, dtype=avg_loss.dtype, device=avg_loss.device)
        avg_loss.update(loss_tensor, weight=torch.ones_like(loss_tensor))

    def compute(self) -> dict[str, float]:
        return {name: cast(Metric, m).compute().item() for name, m in self.metrics.items()}

    def reset(self) -> None:
        for metric in self.metrics.values():
            cast(Metric, metric).reset()
            
        
    def get_metric(self, name: str) -> Metric:
        return cast(Metric, self.metrics[name])
    
    # def print_metrics(self) -> None:
    #     avg_loss_metric = cast(MeanMetric, self.metrics["avg_loss"])
    #     if avg_loss_metric.weight.sum() == 0:
    #         # No updates have been made, skip printing
    #         return 
    #     computed_metrics = self.compute()
    #     for name, value in computed_metrics.items():
    #         # in one line
    #         print(f"{name}: {value:.6f}", end=' | ')
    #     # Ensure the line is finalized; otherwise tqdm may overwrite it.
    #     print(flush=True)
    #     self.reset()