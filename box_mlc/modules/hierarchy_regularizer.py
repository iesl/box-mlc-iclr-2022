"""Structural Regularization for """
from torch.nn.parameter import Parameter
from allennlp.common import Registrable
from allennlp.data.vocabulary import Vocabulary
from pathlib import Path
from networkx.exception import NetworkXException
from typing import List, Tuple, Union, Dict, Any, Optional
import torch
import networkx as nx
import logging
from box_mlc.dataset_readers.hierarchy_readers.hierarchy_reader import (
    HierarchyReader,
)

logger = logging.getLogger(__name__)


class HierarchyRegularizer(torch.nn.Module, Registrable):
    """Base class to satisfy Registrable and to define the common hierarchy initializations"""

    def __init__(
        self,
        alpha: float,
        hierarchy_reader: HierarchyReader,
        debug_level: int = 0,
    ) -> None:
        """
        Args:
            alpha: The regularization parameter that is multiplied with the hierarchy struct loss.
            hierarchy_reader: Creates the adjacency_matrix and the mask
            debug_level: scale of 0 to 3. 0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).


        Returns: (None)

        """
        super().__init__()  # type:ignore
        self.alpha = alpha
        self.debug_level = debug_level
        self.adjacency_matrix = Parameter(
            hierarchy_reader.adjacency_matrix, requires_grad=False
        )  #: Adj(i,j) =1 => if j is true, i is true.
        # self.mask = Parameter(self.initialize_mask(), requires_grad=False) #type: torch.Tensor
        self.mask: torch.BoolTensor = (  # pylint: disable
            hierarchy_reader.mask  # type:ignore
        )  # noqa

    def to(self, *args, **kwargs):  # type: ignore # noqa
        """Deligates to `torch.nn.Module.to`. Additionally moves `self.mask` to the correct device.
        This is needed because we depend on to() to move the all tensors and params to appropriate device.

        Args:
            args: same as super class
            kwargs: same as super class
        """
        super().to(*args, **kwargs)
        (
            device,
            dtype,
            non_blocking,
            convert_to_format,
        ) = torch._C._nn._parse_to(*args, **kwargs)
        self.mask.to(device=device)

    def get_active_adjacency_matrix_and_mask(
        self, active_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            active_mask: 1D Boolean Tensor of shape (adj.shape[0],) indicating which rows and columns to take.

        Returns:
            torch.Tensor: masked adj matrix
            torch.Tensor: masked mask
        """
        assert len(active_mask.shape) == 1
        assert active_mask.shape[0] == self.adjacency_matrix.shape[0]
        num_active = torch.sum(active_mask)
        active_mask_float = active_mask.to(dtype=torch.float)
        active_mask_matrix = torch.ger(
            active_mask_float, active_mask_float
        ).to(dtype=torch.bool)

        return (self.adjacency_matrix[active_mask_matrix]).reshape(
            num_active, num_active
        ), (self.mask[active_mask_matrix]).reshape(num_active, num_active)
