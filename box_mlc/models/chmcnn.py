"""Implements base class for toy vector model"""
from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import (
    Seq2SeqEncoder,
    PassThroughEncoder,
)
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from box_mlc.modules.implication_scorer import (
    ImplicationScorer,
    DotImplicationScorer,
)
from box_mlc.dataset_readers.hierarchy_readers.hierarchy_reader import (
    HierarchyReader,
)
from box_mlc.metrics.mean_average_precision import MeanAvgPrecision
from box_mlc.metrics.f1 import F1
from box_mlc.metrics.micro_average_precision import (
    MicroAvgPrecision,
)
from box_mlc.modules.hierarchy_constraint_violation import (
    ConstraintViolation,
)
from box_mlc.metrics.hierarchy_constraint_violation import (
    ConstraintViolationMetric,
)
from allennlp.training.metrics import Average
from .baseline import BaselineModel
import torch
import logging

logger = logging.getLogger(__name__)


@Model.register("chmcnn")
class CHMCNN(BaselineModel):
    """Does Multilabel classification for toy-data"""

    def __init__(
        self,
        vocab: Vocabulary,
        initializer: InitializerApplicator,
        feedforward: FeedForward,
        hierarchy: HierarchyReader,
        scorer: Optional[ImplicationScorer] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        constraint_violation: Optional[ConstraintViolationMetric] = None,
        warmup_epochs: int = 0,
        batch_regularization: bool = False,
        debug_level: int = 0,
        **kwargs: Any,
    ) -> None:
        """
        Following is the model architecture:

        .. aafig::
            :aspect: 60
            :scale: 150
            :proportional:

            +-------------------------------------------------------+
            |                       +---------+                     |
            |              +------->+  Loss   <--------------+      |
            |              |        +---------+              |      |
            |              |                                 |      |
            |              ^                                 |      |
            |      +-------+------+      +--------------+    |      |
            |      |     Dot      <------+ Label Embed  |    |      |
            |      +-------+------+      +-------^------+    |      |
            |              ^                                 |      |
            |      +-------+------+                          |      |
            |      | feedforward  |                    +-----+      |
            |      +-------^------+                    |            |
            |              |                           |            |
            |              ^                           |            |
            |              +                           +            |
            |   x: List[float]                  Labels: 0,1,0,1,... |
            |                                                       |
            |                                                       |
            |                                                       |
            |                          Single datapoint             |
            |                                                       |
            +-------------------------------------------------------+


        Args:
            vocab: Vocabulary for the model. It will have the following namespaces: labels, tokens, positions
            initializer: Init regexes for all weights in the model. See corresponding `AllenNLP doc <https://docs.allennlp.org/master/api/nn/initializers/#initializerapplicator>`_
            feedforward: Used at the end on either sentence endcoding or mention+sentence encoding based on the concat_mention argument.
            scorer: To score the predicted label representations
            regularizer: See corresponding `AllenNLP doc <https://docs.allennlp.org/master/api/nn/regularizers/regularizer_applicator/>`_
            label_regularizer: Regularization for the relationships in the label space.
            warmup_epochs: Number of epochs to perform warmup training on labels.
                This is only useful when using `label_regularizer` that requires warm-up like `HierarchyRegularizer`. (default:0)
            batch_regularization: Whether to only apply regularization to labels that are seen in the batch.
            debug_level: scale of 0 to 3. 0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).
            **kwargs: Unused
        Returns: (None)

        """
        super().__init__(
            vocab=vocab,
            regularizer=regularizer,
            initializer=initializer,
            feedforward=feedforward,
            scorer=scorer,
            constraint_violation=constraint_violation,
            warmup_epochs=warmup_epochs,
            batch_regularization=batch_regularization,
            debug_level=debug_level,
        )
        self.register_buffer('hierarchy', hierarchy.adjacency_matrix)
        self.hierarchy.requires_grad = False
        self.hierarchy.fill_diagonal_(1.0)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")

    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        """Applies all the regularization for this model.

        Returns:
            `torch.Tensor` containing the total regularization regularization penalty.

        """

        return None

    def get_constr_out(self, x):
        """Given the output of the neural network x returns
        the output of MCM given the hierarchy constraint expressed
        in the matrix R

        See: https://github.com/EGiunchiglia/C-HMCNN/blob/master/main.py#L27

        Ref: Coherent Hierarchical Multi-Label Classification Networks (Neurips 2020)
        """
        R = self.hierarchy
        c_out = x  # (batch, num_labels)
        c_out = c_out.unsqueeze(1)  # (batch, 1, num_labels)
        c_out = c_out.expand(
            len(x), R.shape[1], R.shape[1]
        )  # (batch, num_labels, num_labels)
        R_batch = R.expand(len(x), R.shape[1], R.shape[1])
        final_out, _ = torch.max(R_batch * c_out, dim=-1)

        return final_out

    def forward(  # type:ignore
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        meta: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """

        Args:
            x: Array field Tensors corresponding to the data points.
                Each value tensor will be of shape (batch, x_dim)
            labels: Tensor containing one-hot labels. Has shape (batch, label_set_size).
            meta: Contains raw text and other meta data for each datapoint.
            **kwargs: Unused

        Returns:
            Dict: Tensor dict containing the following keys: loss

        """

        scores = self.get_scores(
            x, labels, meta, **kwargs
        )  # shape (batch, label_set_size)
        # loss
        results: Dict[str, Any] = {"meta": meta}
        positive_probs = torch.sigmoid(scores)
        results["positive_probs"] = positive_probs

        constrained_probs = self.get_constr_out(results["positive_probs"])
        results["positive_probs"] = constrained_probs
        # constrained_probs = self.get_constr_out(scores)

        results["scores"] = constrained_probs

        if labels is not None:
            labels_float = labels.to(dtype=constrained_probs.dtype)
            train_output = ((1.0 - labels_float) * constrained_probs) + (
                labels_float
                * self.get_constr_out(labels_float * positive_probs)
            )
            loss = self.loss_fn(train_output, labels.type_as(train_output))
            results["loss"] = loss

            # metrics
            self.compute_metrics(results, labels)

        if self.epoch < self.warmup_epochs:
            results["loss"] = results["loss"] * 0

        return results
