"""Implements micro average precision using sklearn"""
from typing import List, Tuple, Union, Dict, Any, Optional

from sklearn.metrics import average_precision_score
from allennlp.training.metrics import Metric, Average
import torch


@Metric.register("micro-avg-precision")
class MicroAvgPrecision(Metric):

    """Docstring for MicroAvgPrecision."""

    def __init__(self) -> None:
        super().__init__()
        self.predicted: List[torch.Tensor] = []
        self.gold: List[torch.Tensor] = []

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
        # predictions, gold_labels: (batch_size, labels)
        predictions, gold_labels = [
            t.detach().clone().cpu()
            for t in self.detach_tensors(predictions, gold_labels)
        ]
        self.predicted.append(predictions)
        self.gold.append(gold_labels)

    def get_metric(self, reset: bool) -> float:
        micro_precision_score = -1

        if reset:
            predicted = torch.cat(self.predicted, dim=0)
            gold = torch.cat(self.gold, dim=0)
            labels, scores = [
                t.cpu().numpy() for t in self.detach_tensors(gold, predicted)
            ]
            micro_precision_score = average_precision_score(
                labels, scores, average="micro"
            )

            self.reset()

        return float(micro_precision_score)

    def reset(self) -> None:
        self.predicted = []
        self.gold = []


if __name__ == "__main__":
    map_ = MicroAvgPrecision()

    for i in range(100):
        print(i)
        t = torch.rand(10, 40)
        labels = t > 0.5
        map_(t, labels)
        map_.get_metric(False)
    map_.get_metric(True)
