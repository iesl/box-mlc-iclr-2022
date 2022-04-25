from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from box_embeddings.parameterizations.box_tensor import BoxFactory, BoxTensor
import torch
import logging

logger = logging.getLogger(__name__)


class Vec2Box(torch.nn.Module, Registrable):

    """Docstring for Vec2Box. """

    def __init__(self, box_factory: BoxFactory):
        super().__init__()  # type:ignore
        self.box_factory = box_factory

    def forward(self, vec: torch.Tensor) -> BoxTensor:
        return self.box_factory(vec)


Vec2Box.register("vec2box")(Vec2Box)
