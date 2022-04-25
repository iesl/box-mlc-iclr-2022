"""A box model for toy data multilabel classification"""
from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from box_embeddings.parameterizations.box_tensor import BoxTensor
from box_embeddings.modules.regularization import BoxRegularizer
from allennlp.modules.feedforward import FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from box_embeddings.modules.intersection import Intersection
from box_embeddings.modules.volume import Volume
from box_mlc.modules.hierarchy_regularizer import (
    HierarchyRegularizer,
)
from box_mlc.modules.binary_nll_loss import (
    BinaryNLLLoss,
)
from box_mlc.modules.box_embedding import (
    BoxEmbedding,
    BoxEmbeddingModule,
)
from box_mlc.modules.vector2box import Vec2Box
from box_mlc.metrics.mean_average_precision import (
    MeanAvgPrecision,
)
from box_mlc.metrics.macro_micro_f1 import MicroMacroF1

from box_mlc.modules.hierarchy_constraint_violation import (
    ConstraintViolation,
)
from box_mlc.metrics.hierarchy_constraint_violation import (
    ConstraintViolationMetric,
)
from box_mlc.metrics.f1 import F1
from box_mlc.metrics.micro_average_precision import (
    MicroAvgPrecision,
)
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


@Model.register("mbm")
class ToyBoxModel(Model):
    """Does Multilabel classification for toy-data. But converts the vectors
    into boxes and uses box embedding for labels as well.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        feedforward: FeedForward,
        vec2box: Vec2Box,
        intersect: Intersection,
        volume: Volume,
        label_embeddings: BoxEmbeddingModule,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
        label_box_regularizer: Optional[BoxRegularizer] = None,
        label_regularizer: Optional[HierarchyRegularizer] = None,
        constraint_violation: Optional[ConstraintViolationMetric] = None,
        warmup_epochs: int = 0,
        label_sample_percent: int = 100,
        batch_regularization: bool = False,
        label_loss_sampling: bool = False,
        negative_loss_sampling: bool = False,
        debug_level: int = 0,
        add_new_metrics: bool = True,
        visualization_mode: bool = False,
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
            |     +--------+--------+      +--------------+  |      |
            |     |    Box Volume   +<-----+ Label Embed  |  |      |
            |     +--------+--------+      +-------^------+  |      |
            |              ^                                 |      |
            |     +--------+---------+                       |      |
            |     |   Vector to box  |                       |      |
            |     +--------+---------+                       |      |
            |              ^                                 |      |
            |      +-------+------+                          |      |
            |      |  Vector      |                          |      |
            |      |  Pooler(Avg) |                          |      |
            |      +-------+------+                          |      |
            |              ^                           +-----+      |
            |      +-------+------+                    |            |
            |      | feedforward  |                    |            |
            |      +-------^------+                    |            |
            |        +-----+----+                      |            |
            |        | Dropout  |                      |            |
            |        +-----+----+                      |            |
            |              |                           |            |
            |              ^                           +            |
            |              +                    Labels: 0,1,0,1,... |
            |              x                           |            |
            |                                                       |
            |                          Single datapoint             |
            |                                                       |
            +-------------------------------------------------------+
        Args:
            vocab: Vocabulary for the model. It will have the following namespaces: labels, tokens, positions
            initializer: Init regexes for all weights in the model.
                See corresponding `AllenNLP doc <https://docs.allennlp.org/master/api/nn/initializers/#initializerapplicator>`_
            feedforward: Used at the end on either sentence endcoding or
                mention+sentence encoding based on the concat_mention argument.
            intersect: Box Intersection
            volume: Box Volume
            regularizer: See corresponding
                `AllenNLP doc <https://docs.allennlp.org/master/api/nn/regularizers/regularizer_applicator/>`_
            label_box_regularizer: Applies some regularization to the label embeddings
            label_regularizer: Regularization for the DAG relationships in the label space.
            constraint_violation: to compute the constraint violation if given an hierarchy or hard cooccurence
            warmup_epochs: Number of epochs to perform warmup training on labels.
                This is only useful when using `label_regularizer` that requires warm-up like `HierarchyRegularizer`. (default:0)
            label_sample_percent: Percent of labels to sample for label-label regularization.
                               Default 100 implies include all labels in the regularization loss.
            batch_regularization: Whether to only apply regularization to labels that are seen in the batch.
            label_loss_sampling: Whether to sample labels to compute loss.
            negative_loss_samplingL Whether to sample negative labels to compute main loss
            debug_level: scale of 0 to 3.
                0 meaning no-debug (fastest) and 3 highest debugging possible (slowest).
            visualization_mode: False(default) or True
                Returns the x boxes and y boxes for visualization purposes.
            **kwargs: Unused
        Returns: (None)
        """
        super().__init__(vocab=vocab, regularizer=regularizer)
        self._feedforward = feedforward
        # self._min_ff = FeedForward(self._feedforward.get_output_dim(), 2, [self._feedforward.get_output_dim(), self._feedforward.get_output_dim()//2], activations=[torch.nn.Sigmoid(), Registrable._registry[Activation]['linear'][0]()])
        # self._delta_ff = FeedForward(self._feedforward.get_output_dim(), 2, [self._feedforward.get_output_dim(), self._feedforward.get_output_dim()//2], activations=[torch.nn.Sigmoid(), Registrable._registry[Activation]['linear'][0]()])
        self._volume = volume
        self._intersect = intersect
        self._vec2box = vec2box
        self._label_embeddings = label_embeddings
        self._label_box_regularizer = label_box_regularizer
        self.debug_level = debug_level
        # self._feedforward.should_log_activations=True
        self.visualization_mode = visualization_mode
        self.loss_fn = BinaryNLLLoss()
        self.map = MeanAvgPrecision()
        self._label_regularizer = label_regularizer
        self.epoch = -1  #: completed epochs
        self.warmup_epochs = warmup_epochs
        self.micro_map = MicroAvgPrecision()
        self.f1 = F1()
        self.micro_macro_f1 = MicroMacroF1()

        if add_new_metrics:
            self.micro_map_min_n = MicroAvgPrecision()
            self.map_min_n = MeanAvgPrecision()
            self.micro_map_max_n = MicroAvgPrecision()
            self.map_max_n = MeanAvgPrecision()
        self.add_new_metrics = add_new_metrics
        self.batch_regularization = batch_regularization
        self.label_sample_percent = label_sample_percent
        self.label_loss_sampling = label_loss_sampling
        self.negtive_loss_sampling = negative_loss_sampling
        self.constraint_violation_metric = constraint_violation

        if self.label_loss_sampling and self.negtive_loss_sampling:
            logger.error(
                f"Both label_loss_sampling and negative_loss_sampling cant be true."
            )
            raise ValueError(
                "Both label_loss_sampling and negative_loss_sampling cant be true."
            )

        if warmup_epochs and (label_regularizer is None):
            logger.warning(
                f"Non-zero warmup_epochs ({warmup_epochs})"
                " is not useful when label_regularizer is None"
            )

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

        if initializer is not None:
            initializer(self)

        self.register_buffer(
            "adj",
            torch.tensor(self.constraint_violation_metric.adjacency_matrix),
            persistent=False,
        )

    @torch.no_grad()
    def get_min_normalized_scores(self, scores: torch.Tensor) -> torch.Tensor:
        adj_T = (self.adj.transpose(0, 1))[None, :, :]

        return ((adj_T * scores[:, None, :]).min(dim=-1)[0]).detach()

    @torch.no_grad()
    def get_max_normalized_scores(self, scores: torch.Tensor) -> torch.Tensor:
        adj = (self.adj)[None, :, :]

        return ((adj * scores[:, None, :]).max(dim=-1)[0]).detach()

    def get_device(self):
        for p in self._label_embeddings.parameters():
            return p.device

    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        """Applies all the regularization for this model.
        Returns:
            `torch.Tensor` containing the total regularization regularization penalty.
        """
        penalty = None

        if self._label_box_regularizer is not None:
            penalty = self._label_box_regularizer(
                self._label_embeddings.all_boxes
            )
            # all_boxes = self._label_embeddings.all_boxes
            # penalty = self._label_box_regularizer.weight*torch.sum(all_boxes.Z -all_boxes.z)

        if self._label_embeddings.delta_penalty is not None:
            p = self._label_embeddings.get_delta_penalty()
            penalty = p if penalty is None else penalty + p

        if self._label_regularizer is not None:
            active_nodes: Optional[torch.BoolTensor] = None

            if self.batch_regularization:
                if self.current_labels.numel() == 0:
                    return penalty if penalty is not None else 0.0
                active_nodes = (
                    (self.current_labels).to(dtype=torch.bool).any(dim=0)
                )  # shape (active_nodes,)
                active_boxes = self._label_embeddings(
                    active_nodes.nonzero().flatten()
                )
                target_shape = list(active_boxes.box_shape)
                target_shape.insert(-1, 1)
                reshaped_boxes = active_boxes.box_reshape(
                    tuple(target_shape)
                )  # (active_boxes, -1, box_size)
                label_boxes = active_boxes  # (active_boxes, box_size)
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
                active_nodes = torch.zeros(label_size, device=device).to(
                    dtype=torch.bool
                )
                active_nodes[nodes] = True
                active_boxes = self._label_embeddings(
                    active_nodes.nonzero().flatten()
                )
                target_shape = list(active_boxes.box_shape)
                target_shape.insert(-1, 1)
                reshaped_boxes = active_boxes.box_reshape(
                    tuple(target_shape)
                )  # (label_sample_size, -1, box_size)
                label_boxes = active_boxes  # (label_sample_size, box_size)
            else:
                target_shape = list(self._label_embeddings.all_boxes.box_shape)
                target_shape.insert(-1, 1)
                reshaped_boxes = self._label_embeddings.all_boxes.box_reshape(
                    tuple(target_shape)
                )  # (labels, 1, box_size)
                label_boxes = self._label_embeddings.all_boxes

            label_boxes = label_boxes.box_reshape(
                (1, *label_boxes.box_shape)
            )  # (1, labels, box_size)
            intersection_volume = self._volume(
                self._intersect(reshaped_boxes, label_boxes)
            )  # (labels, labels)

            log_probabilities = intersection_volume - self._volume(
                label_boxes
            )  # shape (labels, labels)

            if (log_probabilities > 0).any():
                logger.warning(
                    f"Label Regularization: {(log_probabilities> 0).sum()} log_probability values greater than 0"
                )
            log_probabilities.clamp_(max=0.0)
            hierarchy_penalty = self._label_regularizer(
                log_probabilities, active_nodes
            )
            penalty = (
                hierarchy_penalty
                if penalty is None
                else penalty + hierarchy_penalty
            )

        return penalty

    def get_scores(  # type:ignore
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        meta: List[Dict[str, Any]],
        results: Dict,
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
            self.current_labels = labels  # hold on the the labels in the buffer for regularization
        encoded_vec = x
        predicted_label_reps = self._feedforward(encoded_vec).unsqueeze(
            -2
        )  # batch, 1, hidden_dims
        # predicted_label_reps = torch.cat((self._min_ff(predicted_label_reps), self._delta_ff(predicted_label_reps)), dim=-1)
        batch, _, hidden_dims = predicted_label_reps.shape

        predicted_label_boxes: BoxTensor = self._vec2box(
            predicted_label_reps
        )  # box_shape (batch, 1, box_size)
        box_size = predicted_label_boxes.box_shape[-1]

        label_boxes: BoxTensor = (
            self._label_embeddings.all_boxes
        )  # box_shape (total_labels, box_size)

        if self.visualization_mode:
            # in vis mode we assume that batch size is 1
            results["x_boxes_z"] = predicted_label_boxes.z.squeeze(
                -2
            )  # Shape: (batch=1, box_size)
            results["x_boxes_Z"] = predicted_label_boxes.Z.squeeze(-2)
            # we need to add one extra dim with size 1 in the begining to fool the
            # forward_on_instances() to think it is batch dim
            results["y_boxes_z"] = label_boxes.z.unsqueeze(
                0
            )  # Shape: (batch=1,labels, box_size)
            results["y_boxes_Z"] = label_boxes.Z.unsqueeze(0)

        total_labels, _ = label_boxes.box_shape
        assert label_boxes.box_shape[1] == box_size
        label_boxes = label_boxes.box_reshape(
            (1, *label_boxes.box_shape)
        )  # box_shape (1, total_labels, box_size)
        assert label_boxes.box_shape == (1, total_labels, box_size)

        intersection = self._intersect(
            predicted_label_boxes, label_boxes
        )  # shape (batch, total_labels, box_size)
        assert intersection.box_shape == (batch, total_labels, box_size)

        log_probabilities = self._volume(intersection) - self._volume(
            predicted_label_boxes
        )  # shape (batch, total_labels)

        if (log_probabilities > 1e-4).any():
            logger.warning(
                f"{(log_probabilities> 0).sum()} log_probability values greater than 0"
            )
        log_probabilities.clamp_(max=0.0)
        assert log_probabilities.shape == (batch, total_labels)

        return log_probabilities

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

        results: Dict[str, Any] = {"meta": meta}
        log_probabilities = self.get_scores(x, labels, meta, results)
        total_labels = log_probabilities.shape[-1]
        # loss
        results["scores"] = log_probabilities
        results["positive_probs"] = torch.exp(log_probabilities)

        if self.debug_level > 0:
            pass
        if labels is not None:
            if self.label_loss_sampling:
                labels_mask = torch.tensor(
                    np.random.choice(
                        [0, 1], size=(total_labels,), p=[0.7, 0.3]
                    ),
                    dtype=torch.bool,
                )
                indices = labels_mask.nonzero().flatten()
                results["loss"] = self.loss_fn(
                    log_probabilities[:, indices], labels[:, indices]
                )
            elif self.negtive_loss_sampling:
                positive_labels = labels.nonzero().flatten()
                negative_labels = list(
                    set(range(total_labels)).difference(set(positive_labels))
                )
                negative_sample_ind = torch.tensor(
                    np.random.choice(
                        negative_labels, size=int(0.3 * len(negative_labels))
                    )
                )
                indices = torch.cat(
                    [positive_labels.cpu(), negative_sample_ind]
                )
                results["loss"] = self.loss_fn(
                    log_probabilities[:, indices], labels[:, indices]
                )
            else:
                results["loss"] = self.loss_fn(log_probabilities, labels)

            if self.epoch < self.warmup_epochs:
                results["loss"] = results["loss"] * 0

            if self.debug_level > 0:
                pass
            # metrics
            self.compute_metrics(results, labels)

            if self.epoch < self.warmup_epochs:
                results["loss"] = results["loss"] * 0

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
        label_boxes = self._label_embeddings.all_boxes
        num_labels, hidden_dims = label_boxes.box_shape
        log_conditional_probs = (
            self._volume(
                self._intersect(
                    label_boxes.box_reshape((1, num_labels, hidden_dims)),
                    label_boxes.box_reshape((num_labels, 1, hidden_dims)),
                )
            )
            - self._volume(label_boxes).unsqueeze(0)
        )

        return log_conditional_probs.cpu().numpy()

    @torch.no_grad()
    def get_marginal(self, label: str) -> float:
        model = self
        try:
            idx = model.vocab.get_token_index(label, namespace="labels")
        except KeyError:
            return None
        box = model._label_embeddings(
            torch.tensor(idx, dtype=torch.long).to(device=self.get_device())
        )
        # compute volume for this model
        vol = model._volume(box)

        return float(vol)
