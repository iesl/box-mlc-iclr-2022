from typing import List
from allennlp.common import Registrable
import numpy as np
import dataclasses
from pathlib import Path
import torch
from allennlp.data import Vocabulary
import sys

if sys.version_info >= (3, 8):
    from typing import (
        TypedDict,
    )  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict, Literal, overload


class HierarchyInfo(TypedDict):
    """Dict that carries information about a hierarchy"""

    adjacency_matrix: torch.Tensor  #: soft edge scores 1 for presence, 0 for absence

    mask: torch.BoolTensor  #: same shape as adjacency_matrix. 0 -> mask, 1 -> unmask


class HierarchyReader(Registrable):
    """Responsible for reading a serialized information about a label-label
    hierarchy and constructing an adjacency_matrix"""

    default_implementation = "networkx-edgelist"

    def __init__(self, vocab: Vocabulary, filepath: Path):
        self.vocab = vocab
        self.filepath = filepath
        self.hierarchy_info = self.read()

    @property
    def adjacency_matrix(self) -> torch.Tensor:
        return self.hierarchy_info["adjacency_matrix"]

    @property
    def mask(self) -> torch.BoolTensor:
        return self.hierarchy_info["mask"]

    def get_all_labels(self) -> List[str]:
        """
        Returns:
            All labels from the vocabulary.
        """

        return list(self.vocab.get_token_to_index_vocabulary("labels").keys())

    def get_label_index(self, label: str) -> int:
        """
        Args:
            label: label name

        Returns:
            The index of the label in the vocabulary
        """

        return self.vocab.get_token_index(label, namespace="labels")

    def check_if_label_present(self, label: str) -> bool:
        return label in self.vocab._token_to_index["labels"]

    def read(self) -> HierarchyInfo:
        """Bulk of the main work has to be done here.

        Raises:
            NotImplementedError: when not implemented in child class
        """
        raise NotImplementedError
