"""
A method that computes the constraint violations, where its considered a
violation if P(General|x) < P(Specific|x)
"""
from typing import Dict, List
from box_mlc.dataset_readers.hierarchy_readers.hierarchy_reader import (
    HierarchyReader,
)
from torch.nn.parameter import Parameter
from allennlp.common import Registrable
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

# TODO: Remove the parent class Module
# TODO: remove the extra useless parameter adjacency_matrix_param
class ConstraintViolation(torch.nn.Module,Registrable):
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
        #self.adjacency_matrix_param = torch.nn.Parameter(hierarchy_reader.adjacency_matrix, requires_grad=False)  # This is useless but present only so that we can load old models.
        self.adjacency_matrix = (
            hierarchy_reader.adjacency_matrix.detach().cpu().numpy()
        )
        self.threshold = cooccurence_threshold

    def get_true_mask(self, true_labels: np.ndarray) -> np.ndarray:
        true_mask = true_labels.copy()
        true_mask[true_mask == 1] = -100000
        true_mask[true_mask == 0] = 1

        return true_mask

    def __call__(
        self, positive_probabilities: torch.Tensor, true_labels: torch.Tensor
    ) -> Dict:
        """
        true_labels: (examples, labels). True labels for the given example.
                    Should follow same label indexing as the adj. matrix.
        positive_probabilities: (examples, labels). Predicted probabilities by the model.
        """
        positive_probabilities = positive_probabilities.detach().cpu().numpy()
        true_labels = true_labels.detach().cpu().numpy()
        edges_idx = np.argwhere(self.adjacency_matrix >= self.threshold)
        true_mask = self.get_true_mask(true_labels)
        # logger.info(f"Processing {len(edges_idx)} edges")
        avg_number_of_violations: List = []
        number_of_violations: List = []
        extent_of_violations: List = []
        frequency: List = []
        distances: List = []
        no_examples_edges_count: int = 0

        for edge in edges_idx:
            ind = np.logical_and(
                true_labels[:, edge[0]], true_labels[:, edge[1]]
            )  # examples where the edge is present
            true_subset = true_labels.copy()[ind]

            if true_subset.shape[0] > 0:
                frequency.append(true_subset.shape[0])
                true_mask_subset = true_mask.copy()[ind]
                true_mask_subset[:, edge[0]] = 1
                true_mask_subset[:, edge[1]] = 1
                positive_subset = positive_probabilities.copy()[
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
                number_of_violations.append(
                    np.sum(np.where(distance_g_s > 0, 1, 0))
                )
                extent_of_violations.append(
                    np.mean(np.maximum(distance_g_s, 0))
                )
                distances.append(np.mean(distance_g_s))
            else:
                no_examples_edges_count += 1
        avg_violation_value = np.sum(
            np.array(avg_number_of_violations) * np.array(frequency)
        ) / np.sum(frequency)
        # logger.info(f'No examples edges count: {no_examples_edges_count}')
        # logger.info(f'Average of avg number of violations: {avg_violation_value}')
        response: Dict = {}
        response["metric_value"] = avg_violation_value
        response[
            "avg_number_of_violations"
        ] = avg_number_of_violations  # List of avergae violations for each edge
        response[
            "number_of_violations"
        ] = number_of_violations  # Number of violations for each edge
        response[
            "distances"
        ] = distances  # list of average distances of G to S in sorgted order of scores
        response["frequency_edges"] = frequency  # frequency of the edges.
        response[
            "extent_of_violations"
        ] = extent_of_violations  # List of difference in value for P(G|X) and P(S|X)

        return response


ConstraintViolation.register("constraint-violation")(ConstraintViolation)
