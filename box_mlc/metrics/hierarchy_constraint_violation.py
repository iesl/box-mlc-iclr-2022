"""
A method that computes the constraint violations, where its considered a
violation if P(General|x) < P(Specific|x)
"""
from typing import Dict, List, Optional, Tuple
from box_mlc.dataset_readers.hierarchy_readers.hierarchy_reader import (
    HierarchyReader,
)
from torch.nn.parameter import Parameter
from allennlp.common import Registrable
from allennlp.training.metrics import Metric
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

# TODO: Remove the parent class Module
# TODO: remove the extra useless parameter adjacency_matrix_param


@Metric.register("slow-hierarchy-constraint-violation")
class SlowConstraintViolationMetric(Metric):
    """
    Given a hierarchy in the form of an adjacency matrix or cooccurence
    statistic in the adjacency matrix format, compute the average
    constraint violation.
    """

    def __init__(
        self,
        hierarchy_reader: HierarchyReader,
        cooccurence_threshold: float = 1,
    ) -> None:
        """
        Args:
            hierarchy_reader: Creates the adjacency_matrix and the mask.
            cooccurence_threshold: If adjecency matrix captures the cooc stats, threshold determines
                                if an edge exixst b/w labels. Row->general.Column->Specific.

        """
        super().__init__()  # type:ignore
        # self.adjacency_matrix_param = torch.nn.Parameter(hierarchy_reader.adjacency_matrix, requires_grad=False)  # This is useless but present only so that we can load old models.
        self.adjacency_matrix = (
            hierarchy_reader.adjacency_matrix.detach().cpu().numpy()
        )
        self.threshold = cooccurence_threshold
        self.positive_probabilities: Optional[np.ndarray] = None
        self.true_labels: Optional[np.ndarray] = None

    def get_true_mask(self, true_labels: np.ndarray) -> np.ndarray:
        true_mask = true_labels.copy()
        true_mask[true_mask == 1] = -100000
        true_mask[true_mask == 0] = 1

        return true_mask

    def __call__(
        self, positive_probabilities: torch.Tensor, true_labels: torch.Tensor
    ) -> None:
        """
        true_labels: (examples, labels). True labels for the given example.
                    Should follow same label indexing as the adj. matrix.
        positive_probabilities: (examples, labels). Predicted probabilities by the model.
        """
        # we are goint to modify scores
        # if it is already on cpu we can to create a copy first
        
        if not positive_probabilities.is_cuda:
            positive_probabilities = positive_probabilities.clone()
        positive_probabilities = positive_probabilities.detach().cpu().numpy()
        true_labels = true_labels.detach().cpu().numpy()

        if self.positive_probabilities is None:
            self.positive_probabilities = positive_probabilities
            self.true_labels = true_labels
        else:
            self.positive_probabilities = np.concatenate(
                (self.positive_probabilities, positive_probabilities), axis=0
            )
            self.true_labels = np.concatenate(
                (self.true_labels, true_labels), axis=0
            )

    def get_metric(self, reset: bool) -> float:

        if reset:
            if self.true_labels is None or self.positive_probabilities is None:
                self.reset()

                return -1
            
            edges_idx = np.argwhere(self.adjacency_matrix >= self.threshold)
            true_mask = self.get_true_mask(self.true_labels)
            # logger.info(f"Processing {len(edges_idx)} edges")
            avg_number_of_violations: List = []
            number_of_violations: List = []
            extent_of_violations: List = []
            frequency: List = []
            distances: List = []
            no_examples_edges_count: int = 0

            for edge in edges_idx:
                ind = np.logical_and(
                    self.true_labels[:, edge[0]], self.true_labels[:, edge[1]]
                )  # examples where the edge is present
                true_subset = self.true_labels.copy()[ind]

                if true_subset.shape[0] > 0:
                    frequency.append(true_subset.shape[0])
                    true_mask_subset = true_mask.copy()[ind]
                    true_mask_subset[:, edge[0]] = 1
                    true_mask_subset[:, edge[1]] = 1
                    positive_subset = self.positive_probabilities.copy()[
                        ind
                    ]  # (#subset_ex, num_labels)
                    extent_of_violations.append(
                        np.mean(
                            positive_subset[:, edge[0]]
                            - positive_subset[:, edge[1]]
                        )
                    )
                    sorted_ind = np.argsort(
                        -1 * positive_subset * true_mask_subset, axis=1
                    )
                    distance_g_s = (
                        np.argwhere(sorted_ind == edge[0])[:, -1]
                        - np.argwhere(sorted_ind == edge[1])[:, -1]
                    )
                    avg_number_of_violations.append(
                        np.sum(np.where(distance_g_s > 0, 1.0, 0.0))
                        / true_subset.shape[0]
                    )

                    if avg_number_of_violations[-1] > 0:
                        breakpoint()
                    number_of_violations.append(
                        np.sum(np.where(distance_g_s > 0, 1, 0))
                    )
                    extent_of_violations.append(
                        np.mean(np.maximum(distance_g_s, 0))
                    )
                    distances.append(np.mean(distance_g_s))
                else:
                    no_examples_edges_count += 1
            metric = float(
                np.sum(
                    np.array(avg_number_of_violations) * np.array(frequency)
                )
                / np.sum(frequency)
            )

            self.reset()
        else:
            metric = -1

        return metric

    def reset(self):
        self.positive_probabilities: Optional[np.ndarray] = None
        self.true_labels: Optional[np.ndarray] = None

def compute_constrain_violations(adjacency_matrix: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    positive_probabilities = scores
    batch, num_labels = positive_probabilities.shape
    adj = np.broadcast_to(
        adjacency_matrix[None, :, :], (batch, num_labels, num_labels)
    )  # (batch, num_labels, num_labels)
    positive_probabilities_diag = positive_probabilities[
        :, :, None
    ]  # (batch, num_labels, 1)
    positive_probabilities_stack = np.broadcast_to(
        positive_probabilities[:, None, :], (batch, num_labels, num_labels)
    )  # (batch, num_labels, num_labels)

    denominator = np.sum(adj)
    numerator = np.sum(
        adj * (positive_probabilities_stack > positive_probabilities_diag)
    )  # violations
    return numerator, denominator


@Metric.register("hierarchy-constraint-violation")
class ConstraintViolationMetric(Metric):
    """
    Given a hierarchy in the form of an adjacency matrix or cooccurence
    statistic in the adjacency matrix format, compute the average
    constraint violation.
    """

    def __init__(
        self,
        hierarchy_reader: HierarchyReader,
    ) -> None:
        """
        Args:
            hierarchy_reader: Creates the adjacency_matrix and the mask.

        """
        super().__init__()  # type:ignore
        # self.adjacency_matrix_param = torch.nn.Parameter(hierarchy_reader.adjacency_matrix, requires_grad=False)  # This is useless but present only so that we can load old models.
        self.adjacency_matrix = (
            hierarchy_reader.adjacency_matrix.detach().cpu().numpy()
        )
        self.numerator: float = 0.0
        self.denominator: float = 0.0

    def __call__(
        self, positive_probabilities: torch.Tensor, true_labels: torch.Tensor
    ) -> None:
        """
        true_labels: (examples, labels). True labels for the given example.
                    Should follow same label indexing as the adj. matrix.
        positive_probabilities: (examples, labels). Predicted probabilities by the model.
        """
        # we are goint to modify scores
        # if it is already on cpu we can to create a copy first

        if not positive_probabilities.is_cuda:
            positive_probabilities = positive_probabilities.clone()
        positive_probabilities = positive_probabilities.detach().cpu().numpy()
        true_labels = true_labels.detach().cpu().numpy()
        batch, num_labels = positive_probabilities.shape
        adj = np.broadcast_to(
            self.adjacency_matrix[None, :, :], (batch, num_labels, num_labels)
        )  # (batch, num_labels, num_labels)
        positive_probabilities_diag = positive_probabilities[
            :, :, None
        ]  # (batch, num_labels, 1)
        positive_probabilities_stack = np.broadcast_to(
            positive_probabilities[:, None, :], (batch, num_labels, num_labels)
        )  # (batch, num_labels, num_labels)

        denominator = np.sum(adj)
        numerator = np.sum(
            adj * (positive_probabilities_stack > positive_probabilities_diag)
        )  # violations
        self.numerator += numerator
        self.denominator += denominator

    def get_metric(self, reset: bool) -> float:
        assert self.denominator > 0
        metric = self.numerator / self.denominator

        if reset:
            self.reset()

        return float(metric)

    def reset(self) -> None:
        self.numerator = 0.0
        self.denominator = 0.0
