from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.training.metrics import Metric
import torch


@Metric.register("micro-macro-f1")
class MicroMacroF1(Metric):

    """Both micro and macro F1."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.tp: Union[float, torch.Tensor] = 0.0
        self.fp: Union[float, torch.Tensor] = 0.0
        self.fn: Union[float, torch.Tensor] = 0.0
        self.eps = 1e-7

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
        # predictions, gold_labels: (batch_size, labels)
        predictions, gold_labels = [
            t.clone() for t in self.detach_tensors(predictions, gold_labels)
        ]
        y_pred = 1.0 * (predictions > self.threshold)
        y_true = gold_labels
        # maintain shape (num_labels, ) for tp, fp and fn
        self.tp += (y_true * y_pred).sum(dim=0)
        self.fp += ((1 - y_true) * y_pred).sum(dim=0)
        self.fn += (y_true * (1 - y_pred)).sum(dim=0)

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        eps = self.eps
        p = self.tp.sum() / (self.tp.sum() + self.fp.sum() + eps)
        r = self.tp.sum() / (self.tp.sum() + self.fn.sum() + eps)
        micro_f1 = 2 * p * r / (p + r + eps)
        macro_p = self.tp / (self.tp + self.fp + eps)
        macro_r = self.tp / (self.tp + self.fn + eps)
        macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r + eps)).mean()

        if reset:
            self.reset()

        return {"micro_f1": float(micro_f1), "macro_f1": float(macro_f1)}

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
