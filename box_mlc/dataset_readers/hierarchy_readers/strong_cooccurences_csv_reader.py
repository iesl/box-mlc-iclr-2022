from typing import List, Tuple, Union, Dict, Any, Optional
from .cooccurences_csv_reader import (
    CooccurrencesCSVReader,
    HierarchyReader,
    HierarchyInfo,
)
import torch
import logging

logger = logging.getLogger(__name__)


@HierarchyReader.register("strong-cooccurrences-csv")
class StrongCooccurrencesCSVReader(CooccurrencesCSVReader):
    """
    Does the same thing as the parent but masks entries more aggressively base on thresholds.

    1. Only keep strong positive and negative correlations

    2. Only keep correlations for labels that occur often
    """

    def __init__(
        self,
        positive_threshold: float,
        presence_percent_threshold: float,
        negative_threshold: float = 0.0,
        **kwargs: Any,
    ):
        """
        Args:
            positive_threshold: Threshold to conclude that A=>B
            presence_percent_threshold: Percentage of total examples that
                should have a particular label for the label to be important enough.
            negative_threshold: Threshold to conclude that A=>~B
            kwargs: Args for the parent class
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.presence_percent_threshold = presence_percent_threshold
        super().__init__(**kwargs)

    def read(self) -> HierarchyInfo:
        hierarchy_info = super().read()
        adj = hierarchy_info["adjacency_matrix"]
        mask_ = hierarchy_info["mask"]
        # breakpoint()
        mask_strong = torch.logical_or(
            adj >= self.positive_threshold, adj <= self.negative_threshold
        )

        occurrence_percent = (
            torch.diagonal(adj) >= self.presence_percent_threshold
        )
        mask_significance = torch.zeros_like(
            mask_, dtype=torch.bool, requires_grad=False
        )
        # make the row and columns for significant labels 1

        for label_id, is_significant in enumerate(occurrence_percent):
            mask_significance[label_id] = is_significant
            mask_significance[:, label_id] = is_significant
        final_mask = torch.logical_and(
            torch.logical_and(mask_significance, mask_strong), mask_
        )

        for i, j in zip(*(torch.nonzero(final_mask, as_tuple=True))):
            i_label = self.vocab.get_token_from_index(
                i.item(), namespace="labels"
            )
            j_label = self.vocab.get_token_from_index(
                j.item(), namespace="labels"
            )
        dependencies = [
            (
                f'{self.vocab.get_token_from_index(i.item(), namespace="labels")}'
                " -> "
                f'{self.vocab.get_token_from_index(j.item(), namespace="labels")}'
                f" with {(adj[i][j]).item()}"
            )
            for i, j in zip(*(torch.nonzero(final_mask, as_tuple=True)))
        ]
        logger.info(
            f"{len(dependencies)} are significant out of {torch.numel(mask_)}"
        )
        logger.debug("Following are the significant dependencies")

        for d in dependencies:
            logger.debug(d)
        # Use thresholds to make solid 1s and 0s
        # make positive
        adj[adj >= self.positive_threshold] = 1.0
        adj[adj <= self.negative_threshold] = 0.0

        return {"adjacency_matrix": adj, "mask": final_mask}  # type: ignore
