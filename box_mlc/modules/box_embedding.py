"""An  `AllenNLP Embedding <https://docs.allennlp.org/v1.0.0/api/modules/token_embedders/embedding/>`_ module which produces boxes instead of vectors"""
from typing import List, Tuple, Union, Dict, Any, Optional, Callable
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.common.registrable import Registrable
from box_embeddings.parameterizations.box_tensor import BoxFactory, BoxTensor
from allennlp.nn.initializers import (
    _InitializerWrapper,
    Initializer,
    PretrainedModelInitializer,
)
from box_embeddings.initializations.uniform_boxes import (
    UniformBoxInitializer as _UniformBoxInitializer,
)
import torch
from allennlp.data.vocabulary import Vocabulary
import logging

logger = logging.getLogger(__name__)


class BoxInitializer(Initializer):
    default_implementation = "uniform-box-initializer"

    def __init__(
        self,
        dimensions: int,
        num_boxes: int = None,
        vocab: Vocabulary = None,
        vocab_namespace: str = "labels",
    ):
        self.dimensions = dimensions

        if num_boxes is None and vocab is None:
            raise ValueError("One of num_boxes or vocab has to be given.")

        if num_boxes is not None:
            self.num_boxes = num_boxes
        else:
            assert vocab is not None
            self.num_boxes = vocab.get_vocab_size(vocab_namespace)
            logger.info(
                f"Using namespace {vocab_namespace} in vocab to find num_boxes"
            )
        logger.info(
            f"Using dimensions={self.dimensions} and num_boxes={self.num_boxes} in box init"
        )

    def __call__(  # type:ignore
        self, box_tensor: torch.Tensor, **kwargs: Any
    ) -> None:
        raise NotImplementedError


@BoxInitializer.register("uniform-box-initializer")
class UniformBoxInitializer(BoxInitializer):
    def __init__(
        self,
        dimensions: int,
        box_type_factory: BoxFactory,
        num_boxes: int = None,
        minimum: float = 0.0,
        maximum: float = 1.0,
        delta_min: float = 0.01,
        delta_max: float = 0.5,
        vocab: Vocabulary = None,
        vocab_namespace: str = "labels",
    ) -> None:
        """

        Args:
            dimensions: TODO
            num_boxes: TODO
            box_type_factory: TODO
            minimum: TODO
            maximum: TODO
            delta_min: TODO
            delta_max: TODO
            vocab: TODO
            vocab_namespace: Namespace in the vocab to use

        Returns: (None)

        """
        super().__init__(
            dimensions, num_boxes, vocab=vocab, vocab_namespace=vocab_namespace
        )
        self._init_func = _UniformBoxInitializer(
            dimensions=self.dimensions,
            num_boxes=self.num_boxes,
            box_type_factory=box_type_factory,
            minimum=minimum,
            maximum=maximum,
            delta_min=delta_min,
            delta_max=delta_max,
        )

    def __call__(self, tensor: torch.Tensor) -> None:  # type:ignore
        self._init_func(tensor)


@Initializer.register("efficient-pretrained")
class EfficientPretrained(PretrainedModelInitializer):
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        super().__call__(*args, **kwargs)
        # remove the stored weights
        del self.weights


@Initializer.register("boxes-from-pretrained-vectors")
class BoxesFromPretrainedVector(PretrainedModelInitializer):
    def __init__(
        self,
        weights_file_path: str,
        box_type_factory: BoxFactory,
        parameter_name_overrides: Dict[str, str] = None,
    ) -> None:
        """

        Args:
            box_type_factory: TODO

        Returns: (None)

        """
        super().__init__(
            weights_file_path,
            parameter_name_overrides=parameter_name_overrides,
        )
        self.box_type_factory = box_type_factory

    def _call_internal(
        self, t: torch.Tensor, source_weights: torch.Tensor
    ) -> None:
        with torch.no_grad():
            box_tensor = self.box_type_factory(source_weights)
            W = box_tensor.W(box_tensor.z, box_tensor.Z, **box_tensor.kwargs)

            if W.shape == t.shape:
                t.copy_(W)
            else:
                emb = box_tensor.zZ_to_embedding(
                    box_tensor.z, box_tensor.Z, **box_tensor.kwargs
                )

                if emb.shape == t.shape:
                    t.copy_(emb)
                else:
                    raise ValueError(
                        f"Shape of weights {t.shape} is not suitable "
                        "for assigning W or embedding"
                    )

    def __call__(self, tensor: torch.Tensor, parameter_name: str, **kwargs) -> None:  # type: ignore
        # Select the new parameter name if it's being overridden

        if parameter_name in self.parameter_name_overrides:
            parameter_name = self.parameter_name_overrides[parameter_name]
        source_weights = self.weights[parameter_name]
        self._call_internal(tensor, source_weights)
        del self.weights


class BoxEmbedding(Embedding):
    default_implementation = "box_embedding"


@BoxEmbedding.register("box_embedding")
class BasicBoxEmbedding(BoxEmbedding):

    """Embedding which returns boxes instead of vectors"""

    def __init__(
        self,
        embedding_dim: int,
        box_factory: BoxFactory = None,
        box_initializer: BoxInitializer = None,
        **kwargs: Any,
    ) -> None:
        box_factory = box_factory or BoxFactory("mindelta_from_vector")
        super().__init__(
            embedding_dim * box_factory.box_subclass.w2z_ratio, **kwargs
        )  # here dim should be (space dim x ratio).
        # we will rename the weight parameter in the parent Embedding
        # class in order to save it from any kind of automatic initializations
        # meant for normal embedding matrix. We have special initialization for
        # box weights.
        # self._parameters["boxweight"] = self._parameters.pop("weight")
        self.box_factory = box_factory

        if box_initializer is None:
            box_initializer = UniformBoxInitializer(
                dimensions=embedding_dim,  # here dim is box dim
                num_boxes=int(self.weight.shape[0]),  # type: ignore
                box_type_factory=self.box_factory,
            )
        logger.info(
            f"Initializing BoxEmbedding boxweight using {box_initializer.__repr__()}"
        )
        box_initializer(self.weight)

    def forward(self, inputs: torch.LongTensor) -> BoxTensor:
        emb = super().forward(inputs)  # shape (..., self.box_embedding_dim*2)
        box_emb = self.box_factory(emb)

        return box_emb

    @property
    def all_boxes(self) -> BoxTensor:
        all_ = self.box_factory(self.weight)  # type:ignore

        return all_

    @property
    def num_boxes(self) -> int:
        return self.weight.shape[0]

    def get_statistics_and_histograms(
        self,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        all_boxes = self.all_boxes
        side_lengths = all_boxes.Z - all_boxes.z  # (num_embeddings, box_size)
        min_side_lengths_per_box = torch.min(side_lengths, dim=-1)[
            0
        ]  # (num_embeddings,)
        max_side_lengths_per_box = torch.max(side_lengths, dim=-1)[0]
        avg_side_lengths_per_box = torch.mean(side_lengths, dim=-1)
        all_side_lengths = torch.flatten(side_lengths)

        return {
            "min_side_length": min_side_lengths_per_box,
            "max_side_length": max_side_lengths_per_box,
            "avg_side_length": avg_side_lengths_per_box,
            "side_lengths": all_side_lengths,
        }

    def get_bounding_box(self) -> BoxTensor:
        all_ = self.all_boxes
        z = all_.z  # shape = (num_embeddings, box_embedding_dim)
        Z = all_.Z
        z_min, _ = z.min(dim=0)
        Z_max, _ = Z.max(dim=0)

        return self.box_factory.box_subclass.from_zZ(z_min, Z_max)


class BoxEmbeddingModule(torch.nn.Module, Registrable):

    """Embedding which returns boxes instead of vectors"""

    default_implementation = "box-embedding-module"

    def __init__(
        self,
        vocab: Vocabulary,
        embedding_dim: int,
        delta_function: str = "softplus",
        delta_penalty: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.center = Embedding(
            embedding_dim, vocab=vocab, vocab_namespace="labels"
        )
        self.delta_by_2 = Embedding(
            embedding_dim, vocab=vocab, vocab_namespace="labels"
        )
        self.delta_penalty = delta_penalty

        if delta_function == "none":
            self.fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
        elif delta_function == "softplus":
            self.fn = torch.nn.Softplus()
        elif delta_function == "square":
            self.fn = lambda x: x ** 2
        elif delta_function == "exp":
            self.fn = lambda x: torch.exp(x)
        else:
            raise ValueError("invalid delta_function")

    def forward(self, inputs: torch.LongTensor) -> BoxTensor:
        center = self.center(inputs)  # shape (..., embedding_dim)
        d2 = self.fn(self.delta_by_2(inputs))  # (..., embeding_dim)
        z = center - d2
        Z = center + d2
        box_emb = BoxTensor.from_zZ(z, Z)

        return box_emb

    def get_delta_penalty(self) -> Optional[torch.Tensor]:

        if self.delta_penalty is not None:
            penalty: Optional[torch.Tensor] = self.delta_penalty * torch.sum(
                self.fn(self.delta_by_2.weight)
            )
        else:
            penalty = None

        return penalty

    @property
    def all_boxes(self) -> BoxTensor:
        center = self.center.weight  # shape (..., embedding_dim)
        d2 = self.fn(self.delta_by_2.weight)  # (..., embeding_dim)
        z = center - d2
        Z = center + d2
        box_emb = BoxTensor.from_zZ(z, Z)

        return box_emb

    @property
    def num_boxes(self) -> int:
        return self.center.weight.shape[0]

    def get_statistics_and_histograms(
        self,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        all_boxes = self.all_boxes
        side_lengths = all_boxes.Z - all_boxes.z  # (num_embeddings, box_size)
        min_side_lengths_per_box = torch.min(side_lengths, dim=-1)[
            0
        ]  # (num_embeddings,)
        max_side_lengths_per_box = torch.max(side_lengths, dim=-1)[0]
        avg_side_lengths_per_box = torch.mean(side_lengths, dim=-1)
        all_side_lengths = torch.flatten(side_lengths)

        return {
            "min_side_length": min_side_lengths_per_box,
            "max_side_length": max_side_lengths_per_box,
            "avg_side_length": avg_side_lengths_per_box,
            "side_lengths": all_side_lengths,
        }

    def get_bounding_box(self) -> BoxTensor:
        all_ = self.all_boxes
        z = all_.z  # shape = (num_embeddings, box_embedding_dim)
        Z = all_.Z
        z_min, _ = z.min(dim=0)
        Z_max, _ = Z.max(dim=0)

        return BoxTensor.from_zZ(z_min, Z_max)


BoxEmbeddingModule.register("box-embedding-module")(BoxEmbeddingModule)
