"""Scorer for x->y i.e score(y|x) implemented as a module"""
from allennlp.common import Registrable
from allennlp.data.vocabulary import Vocabulary
from torch.nn.parameter import Parameter
from typing import List, Tuple, Union, Dict, Any, Optional
import logging
import torch
from torch import linalg as LA
from math import sqrt

logger = logging.getLogger(__name__)


class ImplicationScorer(torch.nn.Module, Registrable):
    """Base class to satisfy Registrable"""

    default_implementation = "dot"

    pass


@ImplicationScorer.register("dot")
class DotImplicationScorer(ImplicationScorer):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__(**kwargs)  # type:ignore

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: yx^T (batch, *, num_labels|out_features) where each scores_yx indicates P(y|x)

        """
        scores = torch.matmul(x, y.T)

        return scores


@ImplicationScorer.register("normalized-dot")
class NormalizedDotImplicationScorer(DotImplicationScorer):
    def __init__(
        self,
        vocab: Vocabulary,
        normalize_length: bool = True,
        scaled_threshold: bool = True,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__(**kwargs)  # type:ignore
        num_labels = vocab.get_vocab_size(namespace="labels")

        if scaled_threshold:
            self.thresholds = torch.nn.Parameter(
                (torch.rand(num_labels) * 10.0)
            )
        else:
            self.thresholds = torch.nn.Parameter((torch.rand(num_labels)))

        self.normalize_length = normalize_length
        self.scaled_threshold = scaled_threshold

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: yx^T (batch, *, num_labels|out_features) where each scores_yx indicates P(y|x)

        """

        if self.normalize_length:
            x_n = x / (
                torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-20
            )  # (batch, *, hidden)
            y_n = y / (torch.norm(y, p=2, dim=-1, keepdim=True) + 1e-20)
        else:
            x_n = x
            y_n = y
        dot = torch.matmul(x_n, y_n.T)  # (batch, *, num_labels)
        leading_dims = x_n.dim() - 1

        if self.scaled_threshold:
            scores = (10.0 * dot) * (
                torch.sigmoid(self.thresholds)[(None,) * leading_dims]
            )
        else:
            scores = self.thresholds[(None,) * leading_dims] * dot

        return scores


@ImplicationScorer.register("bilinear")
class BilinearImplicationScorer(ImplicationScorer):
    def __init__(
        self,
        size: int,
    ) -> None:
        """

        Args:
            size: size of the scorer_weight tensor. Should be hidden_dim.

        Returns: (None)

        """
        super().__init__()  # type:ignore
        self.size = size
        self.scorer_weight = Parameter(
            torch.Tensor(size, size)
        )  # type: torch.Tensor
        torch.nn.init.uniform_(self.scorer_weight, - sqrt(size), sqrt(size))

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features, size)
            x: Shape (batch, *, size)

        Returns:
            scores: xT A y (batch, *, out_features|num_labels), where A = scorer_weight and
                    where each scores_ij represent unnormalized score for P(i|j).

        """
        # weight : (size, size)
        # x: (batch, * , size)
        # y : (num_labels, size)
        scores = torch.matmul(
            torch.matmul(x, self.scorer_weight), y.T
        )  # xT A y

        return scores


@ImplicationScorer.register("hyperbolic")
class HyperbolicImplicationScorer(ImplicationScorer):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__(**kwargs)  # type:ignore
        self.make_asymmetric = False

    def distance(self, y, x):
        """Given the two tensors as input, returns the dot product as the score.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: Eq (1) https://arxiv.org/pdf/2101.04997.pdf
                    (batch, *, num_labels|out_features)

        """
        # y.unsqueeze(0) x.unsqueeze(1) #(1, nl, h) (b, 1, h)
        d = (
            LA.norm(y.unsqueeze(0) - x.unsqueeze(1), dim=-1) ** 2
        )  # batch, num_labels
        denominator_x = 1 - LA.norm(x, dim=-1) ** 2  # batch, *
        denominator_y = 1 - LA.norm(y, dim=-1) ** 2  # num_labels
        denominator = (
            denominator_x.unsqueeze(1) * denominator_y.unsqueeze(0) + 1e-13
        )  # batch, * , num_labels
        d = torch.div(d, denominator)  # batch, *, num_labels
        d = 1 + (2 * d)
        d = torch.where(d > 1.0 + 1e-6, d, torch.ones_like(d) + 1e-6)

        return torch.arccosh(d)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns score based on hyperbolic projection and distance.

        Args:
            y: (out_features|num_labels, in_features|hidden_dim)
            x: Shape (batch, *, in_features|hidden_dim)

        Returns:
            scores: Eq (1) https://arxiv.org/pdf/2101.04997.pdf
                    (batch, *, num_labels|out_features) where each scores_yx indicates P(y|x)
                    where y and x first mapped tp the hyperbolic space, pi(x)=x/(1+(1+||x||^2)^0.5)

        """

        y_denominator = 1 + torch.sqrt(
            1 + LA.norm(y, dim=-1) ** 2
        )  # num_labels
        pi_y = torch.div(
            y, y_denominator.unsqueeze(-1)
        )  # num_labels, hidden_dim
        x_denominator = 1 + torch.sqrt(1 + LA.norm(x, dim=-1) ** 2)  # batch, *
        pi_x = torch.div(
            x, x_denominator.unsqueeze(-1)
        )  # batch, *, hidden_dim
        scores = -self.distance(pi_y, pi_x)  # batch, *,  num_labels

        if self.make_asymmetric:
            general = pi_y
            specific = pi_x
            scores = (
                1
                + 1e-3
                * (
                    torch.linalg.norm(
                        general, ord=2, dim=-1, keepdim=True
                    )  # (num_labels, 1)
                    - torch.linalg.norm(
                        specific, ord=2, dim=-1, keepdim=True
                    ).transpose(0, 1)
                )  # (1, num_labels)
            ) * scores

        return scores


@ImplicationScorer.register("BoxE")
class BoxEScorer(ImplicationScorer):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            **kwargs: TODO

        Returns: (None)

        """
        super().__init__(**kwargs)  # type:ignore

    def convert_to_box_parameters(
        self,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a tensor of shape (*, hidden_dims),
        return the min and delta of shape (*, hidden_dims/2) each.
        """
        len_dim = y.shape[-1]

        if len_dim % 2 != 0:
            raise ValueError(
                f"The last dimension of y should be even but is {y.shape[-1]}"
            )

        split_point = int(len_dim / 2)
        min_coordinate = y.index_select(
            -1,  # dim to split on
            torch.tensor(
                list(range(split_point)),
                dtype=torch.int64,
                device=y.device,
            ),  # index to split on
        )  # (num_labels ,hidden_dim)

        delta = y.index_select(
            -1,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=y.device,
            ),
        )  # (num_labels, hidden_dim)

        return min_coordinate, torch.nn.functional.softplus(delta)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Given the two tensors as input, returns the dot product as the score.

        Note that unlike MBM, for BoxE model, we need to seperate out the label-label scoring from label-input scoring
        because in label-label scoring we will score whether one box is inside the other, while in label-input scoring
        we will need to score whether a point (input) is inside a box.

        Args:
            y: (num_labels, hidden_dim*2), the 2* is for splitting the vector into min/max coordinates of a box.
            x: Shape (batch, hidden_dim) when x is a point, i.e., input and (num_labels, hidden_dims*2) when x is
                the same as y, i.e., we are scoring label-label box containment.

        Returns:
            scores: yx^T (batch, *, num_labels|out_features) where each scores_yx indicates P(y|x)

        """

        if (
            y.shape[0] != x.shape[0]
        ):  # first dim is num_labels and batch, respectively
            min_coordinate, delta = self.convert_to_box_parameters(y)
            min_coordinate = min_coordinate.unsqueeze(
                0
            )  # (1, num_labels, hidden_dim)
            max_coordinate = (
                min_coordinate + delta
            )  # (1, num_labels, hidden_dim)
            center = (
                min_coordinate + delta / 2.0
            )  # (1, num_lables, hidden_dim)
            delta_plus_1 = delta + 1.0  # (1, num_labels, hidden)
            x_reshaped = x.unsqueeze(1)  # (batch , 1, hidden_dim)
            l1_from_center = torch.abs(x_reshaped - center)
            # See: BoxE implementation: https://github.com/ralphabb/BoxE/blob/584c83bb42817c67df0f70c52898aacfc916af19/BoxEModel.py#L68
            # Could not find good justification for such complicated non-smooth distance definition
            # See the original BoxE paper https://proceedings.neurips.cc/paper/2020/hash/6dbbe6abe5f14af882ff977fc3f35501-Abstract.html
            # for a high-level justification.
            per_dim_distance = torch.where(
                torch.logical_and(
                    min_coordinate <= x_reshaped,
                    x_reshaped <= max_coordinate,
                ),  # condition with shape (batch, num_labels, hidden_dim)
                l1_from_center / (delta_plus_1 + 1e-10),
                l1_from_center * delta_plus_1
                - (delta / 2.0 * (delta - 1 / (delta + 1e-10))),
            )  # (batch, num_labels, hidden_dim)
            distance = torch.norm(
                per_dim_distance, dim=-1
            )  # (batch, num_labels)
            scores = -distance
        else:  # both are num_labels
            (
                min_coordinate_parent,
                delta_parent,
            ) = self.convert_to_box_parameters(y)
            min_coordinate_parent = min_coordinate_parent.unsqueeze(
                1
            )  # (num_labels, 1, hidden_dim)
            max_coordinate_parent = (
                min_coordinate_parent + delta_parent
            )  # (num_labels,1, hidden_dim)
            center_parent = (
                min_coordinate_parent + delta_parent / 2.0
            )  # (num_lables, 1, hidden_dim)
            delta_plus_1_parent = delta_parent + 1.0  # (num_labels, 1, hidden)

            (
                min_coordinate_child,
                delta_child,
            ) = self.convert_to_box_parameters(x)
            min_coordinate_child = min_coordinate_child.unsqueeze(
                0
            )  # (1, num_labels, hidden_dim)
            max_coordinate_child = (
                min_coordinate_child + delta_child
            )  # (1, num_labels, hidden_dim)
            center_child = (
                min_coordinate_child + delta_child / 2.0
            )  # (1, num_lables, hidden_dim)
            delta_plus_1_child = delta_child + 1.0  # (1, num_labels, hidden)
            # Again from the paper, it is not clear HOW to enforce the containment because the exact equations for the loss are not given!!
            # So we try our best to formulate a reasonable way to do it.
            min_coordinate_containment = torch.where(
                min_coordinate_parent <= min_coordinate_child,
                torch.zeros_like(min_coordinate_child),
                torch.abs(min_coordinate_parent - min_coordinate_child),
            )  # (num_labels, num_labels, hidden_dim)
            max_coordinate_containment = torch.where(
                max_coordinate_parent >= max_coordinate_child,
                torch.zeros_like(max_coordinate_child),
                torch.abs(max_coordinate_parent - max_coordinate_child),
            )  # (num_labels, num_labels, hidden_dim)

            per_dim_containment_violations = (
                min_coordinate_containment + max_coordinate_containment
            )
            scores = -torch.norm(per_dim_containment_violations, dim=-1)

        return scores
