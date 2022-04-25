from typing import Optional, Set
import numpy as np
from pathlib import Path
import torch
from allennlp.data import Vocabulary
from .hierarchy_reader import HierarchyReader, HierarchyInfo
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@HierarchyReader.register("networkx-edgelist")
class NetworkXEdgeListReader(HierarchyReader):
    def __init__(
        self,
        vocab: Vocabulary,
        filepath: Path,
        delimiter: Optional[str] = None,
        symmetric: bool = False,
        include_negative_edges_mask: bool = False,
    ):
        """

        Args:
            vocab: Model's vocab. No passed in config.
            filepath: path to the edgelist file
            delimiter: Delimiter used in the edgelist file
        """
        self.delimiter = delimiter
        self.symmetric = symmetric
        self.include_negative_edges_mask = include_negative_edges_mask
        self.warned_nodes: Set[str] = set()

        # one of the rare class hierarchies where super constructor should be called after.
        super().__init__(vocab, filepath)

    def get_label_graph(self) -> nx.DiGraph:
        """
        Returns:
            Graph constructed from the edgelist
        """

        if self.delimiter:
            label_graph = nx.read_edgelist(
                self.filepath,
                create_using=nx.DiGraph,
                delimiter=self.delimiter,
            )
        else:
            label_graph = nx.read_edgelist(
                self.filepath, create_using=nx.DiGraph
            )

        return label_graph

    def read(self) -> HierarchyInfo:
        return {
            "adjacency_matrix": self.get_adjaceny_matrix(),
            "mask": self.get_mask(),
        }

    def get_adjaceny_matrix(self) -> torch.Tensor:
        """
        Adjacency matrix is initialized to represent the hierarchial relationship in the label space.
        Here the structure is defined by the edgelist file. We try to capture the probability of an
        edge existing between two labels in this list such that adjacency_matrix[i][j]=1 implies an edge
        from i to j where i represents a general label and j represents a more specific label and
        P(general label| specific label)=1.

        However, the dataset has inverted representation where an edge goes from a specific label to
        a generic one. So we transpose the matrix in the initialization to capture the correct probabilities
        to be used for computing the hierarchial regularization loss.

        Returns:
            Adjacency matrix of size: (label_size, label_size).
        """
        all_labels = self.get_all_labels()
        label_size = len(all_labels)
        label_graph = self.get_label_graph()
        adjacency_matrix = torch.zeros(label_size, label_size)
        adjacency_matrix.fill_diagonal_(1.0)

        for label in all_labels:
            label_idx = self.get_label_index(label)

            try:
                out_edges = label_graph.out_edges(label)
            except nx.exception.NetworkXException as e:
                # Happens when there are no outgoing edges from this node
                # ie. it is a single node or all the edges come in.
                # We do not need to do anything. The column corresponding to this node will be 0.
                logger.error(
                    f"Exception ({e}) occured retreiving outedges for {label} from label graph"  # type:ignore
                )

                continue

            for label_specific, label_general in out_edges:
                if self.check_if_label_present(label_general):
                    general_idx = self.get_label_index(label_general)
                elif label_general not in self.warned_nodes:
                    logger.warning(
                        f"Label({label_general}) from hierarchy doesnt exist in vocabulary"  # type:ignore
                    )
                    self.warned_nodes.add(label_general)

                    continue
                else:
                    continue
                adjacency_matrix[general_idx][label_idx] = 1

                if self.symmetric:
                    adjacency_matrix[label_idx][general_idx] = 1

        return adjacency_matrix

    def get_mask_including_negative_edges(self) -> torch.BoolTensor:
        """
        Hierarchial regularization loss(HRL) is an additional signal to the model for learning the
        hierarchial relationships that exist in the label space. A mask is created (as follows) based on
        what sort of relationships contributes towards the loss:
        (a) If there is an edge existing between a label(general) to another label(specific),
            then P(general|specific) should be 1. This contributes towards the HRL. Mask at this index should
            be 1.
        (b) However, in continuation to point(a), P(specific|general) can be anything. This does not contribute to HRL
            and mask should be 0. (unsure about this)
        (c) P(label|label) can be anything based on the method to compute probability. Mask at diagonals should be 0.
        (d) If there exists no edge between two labels, then P(label1|label2)=P(label2|label1)=0. Mask here should be 1.

        Returns:
            Mask of size: (label_size, label_size).
        """
        logger.info(
            "Alternate get_mask called: includes negative edges in the mask"  # type:ignore
        )
        label_graph = self.get_label_graph()
        all_labels = self.get_all_labels()
        label_size = len(all_labels)
        mask = torch.ones(  # type:ignore
            label_size, label_size, dtype=bool, requires_grad=False
        )

        for idx in range(label_size):
            mask[idx][idx] = False  # point (c)

        for specific, general in label_graph.edges():
            # Note that the dataset has edge from specific(u) to general(v).

            if self.check_if_label_present(specific):
                specific_idx = self.get_label_index(specific)
            else:
                if specific not in self.warned_nodes:
                    logger.warning(
                        f"Label({specific}) from hierarchy doesnt exist in vocabulary"  # type:ignore
                    )
                    self.warned_nodes.add(specific)

                continue

            if self.check_if_label_present(general):
                general_idx = self.get_label_index(general)
            else:
                if general not in self.warned_nodes:
                    logger.warning(
                        f"Label({general}) from hierarchy doesnt exist in vocabulary"  # type:ignore
                    )
                self.warned_nodes.add(general)

                continue
            mask[specific_idx][general_idx] = False  # point (b) changed

        return mask

    def get_mask(self) -> torch.BoolTensor:
        """
        Hierarchial regularization loss(HRL) is an additional signal to the model for learning the
        hierarchial relationships that exist in the label space. A mask is created (as follows) based on
        what sort of relationships contributes towards the loss:
        (a) If there is an edge existing between a label(general) to another label(specific),
            then P(general|specific) should be 1. This contributes towards the HRL. Mask at this index should
            be 1.
        (b) However, in continuation to point(a), P(specific|general) can be anything. This does not contribute to HRL
            and mask should be 0. (unsure about this)
        (c) P(label|label) can be anything based on the method to compute probability. Mask at diagonals should be 0.
        (d) If there exists no edge between two labels, then P(label1|label2)=P(label2|label1)=0. Mask here should be 1.

        Returns:
            Mask of size: (label_size, label_size).
        """

        if self.include_negative_edges_mask:
            return self.get_mask_including_negative_edges()

        label_graph = self.get_label_graph()
        all_labels = self.get_all_labels()
        label_size = len(all_labels)
        mask = torch.zeros(  # type:ignore
            label_size, label_size, dtype=bool, requires_grad=False
        )

        for specific, general in label_graph.edges():
            # Note that the dataset has edge from specific(u) to general(v).

            if self.check_if_label_present(specific):
                specific_idx = self.get_label_index(specific)
            else:
                if specific not in self.warned_nodes:
                    logger.warning(
                        f"Label({specific}) from hierarchy doesnt exist in vocabulary"  # type:ignore
                    )
                    self.warned_nodes.add(specific)

                continue

            if self.check_if_label_present(general):
                general_idx = self.get_label_index(general)
            else:
                if general not in self.warned_nodes:
                    logger.warning(
                        f"Label({general}) from hierarchy doesnt exist in vocabulary"  # type:ignore
                    )
                    self.warned_nodes.add(general)

                continue
            mask[general_idx][specific_idx] = True
            mask[specific_idx][general_idx] = True

        return mask
