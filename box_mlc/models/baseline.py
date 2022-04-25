"""Implements base class for toy vector model"""
from typing import List, Tuple, Union, Dict, Any, Optional
import numpy as np
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
    HyperbolicImplicationScorer,
    BoxEScorer,
)
from box_mlc.metrics.macro_micro_f1 import MicroMacroF1
from box_mlc.modules.binary_nll_loss import (
    BinaryNLLLoss,
    MarginLoss,
)
from box_mlc.modules.hierarchy_regularizer import (
    HierarchyRegularizer,
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
import torch
import logging

logger = logging.getLogger(__name__)


@Model.register("baseline")
class BaselineModel(Model):
    """Does Multilabel classification for toy-data"""

    def __init__(
        self,
        vocab: Vocabulary,
        feedforward: FeedForward,
        scorer: Optional[ImplicationScorer] = None,
        regularizer: Optional[RegularizerApplicator] = None,
        label_regularizer: Optional[HierarchyRegularizer] = None,
        label_sample_percent: int = 100,
        constraint_violation: Optional[ConstraintViolationMetric] = None,
        warmup_epochs: int = 0,
        batch_regularization: bool = False,
        debug_level: int = 0,
        initializer: Optional[InitializerApplicator] = None,
        binary_nll_loss: bool = False,
        add_new_metrics: bool = True,
        box_e: bool = False,
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
            label_sample_percent: Percent of labels to sample for label-label regularization.
                               Default 100 implies include all labels in the regularization loss.
            debug_level: scale of 0 to 3. 0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).
            **kwargs: Unused
        Returns: (None)

        """
        super().__init__(vocab=vocab, regularizer=regularizer)
        self._feedforward = feedforward
        self._label_embeddings = torch.nn.Embedding(
            vocab.get_vocab_size(namespace="labels"),
            self._feedforward.get_output_dim() if not box_e else self._feedforward.get_output_dim() * 2,  # type: ignore
        )
        self._label_regularizer = label_regularizer
        self.debug_level = debug_level
        self.constraint_violation = constraint_violation
        self.binary_nll_loss = binary_nll_loss
        self.box_e = box_e

        if box_e:
            self.loss_fn = MarginLoss(reduction="mean")
        elif binary_nll_loss:
            self.loss_fn = BinaryNLLLoss(reduction="mean")
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

        self.map = MeanAvgPrecision()

        if box_e:
            self.scorer = BoxEScorer()
        else:
            self.scorer = scorer or DotImplicationScorer()
        self.warmup_epochs = warmup_epochs
        self.label_sample_percent = label_sample_percent
        self.micro_map = MicroAvgPrecision()
        self.f1 = F1()
        self.micro_macro_f1 = MicroMacroF1()

        if add_new_metrics:
            self.micro_map_min_n = MicroAvgPrecision()
            self.map_min_n = MeanAvgPrecision()
            self.micro_map_max_n = MicroAvgPrecision()
            self.map_max_n = MeanAvgPrecision()
        self.add_new_metrics = add_new_metrics
        self.constraint_violation_metric = constraint_violation

        if warmup_epochs and (label_regularizer is None):
            logger.warning(
                f"Non-zero warmup_epochs ({warmup_epochs})"
                " is not useful when label_regularizer is None"
            )
        self.batch_regularization = batch_regularization

        if label_regularizer is None and (
            label_sample_percent <= 0 or label_sample_percent > 100
        ):
            logger.error(
                f"Invalid label_sample_percent value: {label_sample_percent}"
            )
            raise ValueError(
                f"Invalid label_sample_percent value: {label_sample_percent}"
            )

        self.register_buffer(
            "current_labels", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "adj",
            torch.tensor(self.constraint_violation_metric.adjacency_matrix),
            persistent=False,
        )
        self.epoch = -1  #: completed epochs

        if initializer is not None:
            initializer(self)

    def get_device(self):
        for p in self._label_embeddings.parameters():
            return p.device

    @torch.no_grad()
    def get_min_normalized_scores(self, scores: torch.Tensor) -> torch.Tensor:
        adj_T = (self.adj.transpose(0, 1))[None, :, :]

        return ((adj_T * scores[:, None, :]).min(dim=-1)[0]).detach()

    @torch.no_grad()
    def get_max_normalized_scores(self, scores: torch.Tensor) -> torch.Tensor:
        adj = (self.adj)[None, :, :]

        return ((adj * scores[:, None, :]).max(dim=-1)[0]).detach()

    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        """Applies all the regularization for this model.

        Returns:
            `torch.Tensor` containing the total regularization regularization penalty.

        """

        if self._label_regularizer is not None:
            active_nodes: Optional[torch.BoolTensor] = None

            if self.batch_regularization:
                if self.current_labels.numel() == 0:
                    return 0.0  # this happens in the a call before training starts
                active_nodes = (
                    (self.current_labels).to(dtype=torch.bool).any(dim=0)
                )
                active_nodes_ids = active_nodes.nonzero().flatten()
                active_embeddings = self._label_embeddings(active_nodes_ids)
                scores = self.scorer(active_embeddings, active_embeddings)
            elif self.label_sample_percent != 100:
                # (label_sample_size, )
                label_size = self.vocab.get_vocab_size(namespace="labels")
                label_sample_size = int(
                    (label_size * self.label_sample_percent) / 100
                )
                device = self.get_device()
                nodes = torch.randperm(label_size, device=device)[
                    :label_sample_size
                ]
                active_embeddings = self._label_embeddings(nodes)
                active_nodes = torch.zeros(label_size, device=device).to(
                    dtype=torch.bool
                )
                active_nodes[nodes] = True
                scores = self.scorer(active_embeddings, active_embeddings)
            else:
                scores = self.scorer(
                    self._label_embeddings.weight,
                    self._label_embeddings.weight,
                )

            return self._label_regularizer(scores, active_nodes)
        else:
            return None

    def get_scores(  # type:ignore
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        meta: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> torch.Tensor:
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

        if labels is not None:
            self.current_labels = labels
        encoded_vec = x
        predicted_label_reps = self._feedforward(encoded_vec)  # b, hidden_size
        scores = self.scorer(
            self._label_embeddings.weight, predicted_label_reps
        )  # shape (batch, label_set_size)

        return scores

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
        results["scores"] = scores
        results["positive_probs"] = (
            torch.exp(scores)
            if self.binary_nll_loss
            else torch.sigmoid(scores)
        )
        results["loss"] = self.loss_fn(scores, labels.type_as(scores))

        if self.epoch < self.warmup_epochs:
            results["loss"] = results["loss"] * 0

        # metrics

        if labels is not None:
            self.compute_metrics(results, labels)

        return results

    def compute_metrics(self, results: Dict, labels: torch.Tensor) -> None:
        self.map(results["scores"], labels)
        self.f1(results["positive_probs"], labels)
        self.micro_map(results["scores"], labels)
        self.micro_macro_f1(results["positive_probs"], labels)

        if self.constraint_violation_metric is not None:
            self.constraint_violation_metric(results["scores"], labels)

        if self.add_new_metrics:
            s = self.get_min_normalized_scores(results["scores"])
            p = self.get_min_normalized_scores(results["positive_probs"])
            self.map_min_n(s, labels)
            self.micro_map_min_n(s, labels)
            s = self.get_max_normalized_scores(results["scores"])
            p = self.get_max_normalized_scores(results["positive_probs"])
            self.map_max_n(s, labels)
            self.micro_map_max_n(s, labels)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "MAP": self.map.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            "micro_map": self.micro_map.get_metric(reset),
            **self.micro_macro_f1.get_metric(reset),
        }

        if self.constraint_violation_metric is not None:
            metrics[
                "constraint_violation"
            ] = self.constraint_violation_metric.get_metric(reset)

        if self.add_new_metrics:
            metrics = {
                **metrics,
                **{
                    "MAP_min_n": self.map_min_n.get_metric(reset),
                    "micro_map_min_n": self.micro_map_min_n.get_metric(reset),
                    # **{
                    #    f1_str + "_min_n": f1_
                    #    for f1_str, f1_ in self.micro_macro_f1_min_n.get_metric(
                    #        reset
                    #    ).items()
                    # }
                    # "micro_map": self.micro_map.get_metric(reset),
                },
                **{
                    "MAP_max_n": self.map_max_n.get_metric(reset),
                    "micro_map_max_n": self.micro_map_max_n.get_metric(reset),
                    # "fixed_f1_max_n": self.f1_max_n.get_metric(reset),
                    # **{
                    #    f1_str + "_max_n": f1_
                    #    for f1_str, f1_ in self.micro_macro_f1_max_n.get_metric(
                    #        reset
                    #    ).items()
                    # }
                    # "micro_map": self.micro_map.get_metric(reset),
                },
            }

        return metrics

    def make_output_human_readable(  # type: ignore
        self, output_dict: Dict[str, Union[torch.Tensor, Dict]]
    ) -> Dict[str, Union[torch.Tensor, List]]:
        threshold = 0.5
        batch_size = output_dict["positive_probs"].shape[0]  # type:ignore
        preds_list_idx: List[List[str]] = [
            (
                (output_dict["positive_probs"][sample_number] > threshold)
                .nonzero()
                .view(-1)
                .tolist()
            )
            for sample_number in range(batch_size)
        ]
        preds = (
            output_dict["positive_probs"] >= threshold  # type: ignore
        )  # shape (batch, label_set_size)

        output_dict["predictions"] = [
            [  # type: ignore
                self.vocab.get_index_to_token_vocabulary(  # type: ignore
                    namespace="labels"
                ).get(pred_idx)
                for pred_idx in example_pred_list
            ]
            for example_pred_list in preds_list_idx
        ]

        return output_dict  # type: ignore

    @torch.no_grad()
    def get_edge_scores(self) -> np.ndarray:
        if isinstance(self.scorer, HyperbolicImplicationScorer):
            self.scorer.make_asymmetric = True
            specific = self._label_embeddings.weight
        elif isinstance(self.scorer, BoxEScorer):
            self.scorer.make_asymmetric = False
            specific = self._label_embeddings.weight
        else:
            specific = self._label_embeddings.weight / torch.linalg.norm(
                self._label_embeddings.weight, ord=2, dim=-1, keepdim=True
            )
        edge_scores = self.scorer(self._label_embeddings.weight, specific)
        # edge_scores = self.scorer(
        #    self._label_embeddings.weight, self._label_embeddings.weight
        # )

        return edge_scores

    @torch.no_grad()
    def get_marginal(self, label: str) -> float:
        model = self
        try:
            idx = model.vocab.get_token_index(label, namespace="labels")
        except KeyError:
            return None
        label_emb = model._label_embeddings(
            torch.tensor(idx, dtype=torch.long).to(device=self.get_device())
        )
        # compute volume for this model
        magnitude = torch.linalg.norm(label_emb, ord=2)

        if isinstance(self.scorer, HyperbolicImplicationScorer):
            magnitude = -magnitude

        return float(magnitude)
