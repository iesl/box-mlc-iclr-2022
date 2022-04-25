"""Structural Regularization for """
from box_mlc.modules.hierarchy_regularizer import (
    HierarchyRegularizer,
)
from box_mlc.modules.binary_nll_loss import (
    BinaryNLLLoss,
)
from allennlp.common import Registrable
from allennlp.data.vocabulary import Vocabulary
from pathlib import Path
from torch.nn.parameter import Parameter
from typing import List, Tuple, Union, Dict, Any, Optional
import torch
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@HierarchyRegularizer.register("normalized-input")
class NormalizedHierarchyRegularizer(HierarchyRegularizer):
    def __init__(self, **kwargs: Any) -> None:
        """

        Args:
            kwargs: All the parameters same as for the parent class
        """
        super().__init__(**kwargs)
        self.loss_function = BinaryNLLLoss(reduction="none")

    def forward(
        self,
        scores: torch.Tensor,
        active_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            scores: Size (label_size, label_size) or (active_label_size, active_label_size).
                    Normalized scores(log probabilities) for an edge between two labels.
            active_mask: 1D Bool Tensor of indicating which rows and columns to take

        Returns:
            Scaled Binary Negative Log Likelihood Loss.
        """

        if active_mask is not None:
            adjacency_matrix, mask = self.get_active_adjacency_matrix_and_mask(
                active_mask
            )
        else:
            adjacency_matrix, mask = self.adjacency_matrix, self.mask

        if len(torch.nonzero(mask)) == 0:
            return torch.tensor(0.0)

        masked_scores: torch.Tensor = scores[mask].unsqueeze(
            -1
        )  # (N, 1) where N is +ve-mask values
        masked_true_prob: torch.Tensor = adjacency_matrix[mask].unsqueeze(
            -1
        )  # (N, 1)
        loss = torch.sum(
            self.loss_function(
                masked_scores, masked_true_prob.long()
            )  # (total_labels, total_labels,1)
        )

        return self.alpha * loss
