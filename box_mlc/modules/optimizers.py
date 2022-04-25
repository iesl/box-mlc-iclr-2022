from typing import List, Tuple, Union, Dict, Any, Optional
from torch_optimizer import Yogi
from allennlp.training.optimizers import Optimizer, make_parameter_groups
import torch


@Optimizer.register("yogi")
class YogiOptimizer(Optimizer, Yogi):
    """
    Registered as an `Optimizer` with name "yogi".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        initial_accumulator: float = 1e-6,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            initial_accumulator=initial_accumulator,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
