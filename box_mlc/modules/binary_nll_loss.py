"""Implements Negative Log Likelihood Loss for multi instance typing model"""
from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common import Registrable
from box_embeddings.common.utils import log1mexp
import logging
import torch

logger = logging.getLogger(__name__)


class Sampler(Registrable):
    """Given target binary vector of shape (batch, total_labels) and predicted log probabilities of shape (batch, total_labels)
    performs sampling to produce a sampled target of shape (batch, sample_size) and log probabilities of shape (batch, sample_size)
    """

    default_implementation = "identity"

    def sample(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return scores, targets, None


Sampler.register("identity")(Sampler)


@Sampler.register("true-negatives")
class TrueNegativeSampler(Sampler):
    def __init__(
        self, number_of_negatives: int = 1, adversarial: bool = False
    ):
        self.number_of_negatives = number_of_negatives
        self.adversarial = adversarial

    def sample(
        self, scores: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        targets_float = targets.to(dtype=torch.float)

        neg_distribution = 1.0 - targets_float
        positive_distribution = targets_float

        if self.adversarial:
            neg_distribution = torch.softmax(scores, dim=-1) * neg_distribution
            # positive_distribution = (
            # torch.softmax(-scores, dim=-1) * targets_float
            # )
        positives_sample_indices = torch.multinomial(
            positive_distribution, 1
        )  # (batch, 1)
        negative_sample_indices = torch.multinomial(
            neg_distribution, self.number_of_negatives
        )  # (batch, number_of_negatives)
        # (this can generate false negatives if number_of_negatives > true negatives for an instance.
        # However, that is very unlikely. So we will accept this.
        positives_sample = torch.gather(
            scores, -1, positives_sample_indices
        )  # (batch, 1)
        negative_sample = torch.gather(
            scores, -1, negative_sample_indices
        )  # (batch, number_of_negative)
        sample_scores = torch.cat(
            (positives_sample, negative_sample), dim=-1
        )  # (batch, 1+num_negatives)
        sample_targets = torch.cat(
            (
                torch.gather(targets, -1, positives_sample_indices),
                torch.gather(targets, -1, negative_sample_indices),
            ),
            dim=-1,
        )
        weights = torch.tensor([1.0, self.number_of_negatives]).to(
            device=sample_scores.device, dtype=sample_scores.dtype
        )

        return sample_scores, sample_targets, weights


class BinaryNLLLoss(torch.nn.Module, Registrable):
    """Given log P of the positive class, computes the NLLLoss using log1mexp for log 1-P."""

    default_implementation = "binary-nllloss"

    def __init__(
        self,
        debug_level: int = 0,
        reduction: "str" = "mean",
        sampler: Sampler = None,
    ) -> None:
        """
        Args:
            debug_level: scale of 0 to 3.
                0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).
            reduction: Same as `torch.NLLLoss`.
            **kwargs: Unused
        Returns: (None)
        """
        super().__init__()
        self.debug_level = debug_level
        self.reduction = reduction
        self.sampler = sampler or Sampler()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input: (predicted_log_probabilities) log probabilities for each label predicted by the model.
                                         Size is (batch, total_labels)
            target: (true_labels) True labels for the data (batch, total_labels)
        Returns:
            The negative log likelihood loss between the predicted and true labels.
        """
        weights = None

        if self.training:  # only sampler during training
            input, target, weights = self.sampler.sample(input, target)
        log1mexp_log_probabilities = log1mexp(input)

        if self.debug_level > 0:
            pass
        predicted_prob = torch.stack(
            [log1mexp_log_probabilities, input], -2
        )  # (batch, 2, total_labels)
        loss = torch.nn.functional.nll_loss(
            predicted_prob,
            target.to(dtype=torch.long),
            weight=weights,
            reduction=self.reduction,
        )  # (batch, total_labels)

        return loss

    def reduce_loss(self, loss_tensor: torch.Tensor) -> None:
        """
        Args:
            loss_tensor: Computed loss values (batch, total_labels)
        Returns:
            Reduced value by summing up the values across labels dimension
            and averaging across the batch dimension (torch.Tensor)
        """
        # return torch.mean(torch.sum(torch.topk(loss_tensor,500,dim=-1,sorted=False)[0], -1))

        return torch.mean(torch.sum(loss_tensor, -1))


BinaryNLLLoss.register("binary-nllloss")(BinaryNLLLoss)


class MarginLoss(torch.nn.Module, Registrable):
    """Margin based loss of multi-label """

    def __init__(
        self,
        debug_level: int = 0,
        reduction: "str" = "mean",
    ) -> None:
        """
        Args:
            debug_level: scale of 0 to 3.
                0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).
            reduction: Same as `torch.NLLLoss`.
            **kwargs: Unused
        Returns: (None)
        """
        super().__init__()
        self.debug_level = debug_level
        self.reduction = reduction
        self.loss_fn = torch.nn.MultiLabelMarginLoss(reduction=reduction)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input:  predicted scores for each label by the model.
                                         Size is (batch, total_labels)
            target: (true_labels) True labels for the data (batch, total_labels)
        Returns:
            The margin loss comparing each positive with each negative.
        """
        # convert targets to the form expected by multilabelmarginloss
        batch, num_labels = target.shape
        targets_reformated = (
            torch.sort(
                (
                    torch.arange(
                        1, num_labels + 1, device=target.device
                    ).unsqueeze(0)
                    * target
                ),
                dim=-1,
                descending=True,
            )[0]
            - 1
        )
        loss = self.loss_fn(input, targets_reformated.to(dtype=torch.long))

        return loss
