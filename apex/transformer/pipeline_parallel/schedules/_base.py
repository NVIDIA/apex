import abc
from dataclasses import dataclass
from typing import List, Union, Optional

import torch

from apex.transformer.pipeline_parallel.schedules.common import Batch, FwdStepFunc


# TODO (mkozuki): Rewrite "forward_backward" functions using this class to reduce cyclomatic complexity.
@dataclass
class ForwardBackwardBase():
    forward_step_func: FwdStepFunc
    batch: Union[Batch, List[Batch]]
    model: Union[torch.nn.Module, List[torch.nn.Module]]
    forward_only: bool
    tensor_shape: Optional[Union[List[int], torch.Size]] = None

    """
    Base class for pipeline parallelism applied forward & backward path.

    `__call__` method calls three member methods, (1) run_warmup, (2)  run_steady_state, (3) run_cooldown.

    -----------------------------------
                    |
                    |
        --------------------------
        |       warmup           |
        --------------------------
                    |
                    |
        --------------------------
        |    steady state        |
        -------------------------
                    |
                    |
        -------------------------
        |      cooldown         |
        ------------------------
                    |
                    |
    ------------------------------------

    Args:
        forward_step_func: A function which takes a minibatch and model as its arguments and
            returns model's forward output and the loss function.
            The loss function is supposed to take one `torch.Tensor` and
            return a `torch.Tensor` of loss and a dictionary of `str` and `torch.Tensor`.
        batch: A minibatch, i.e., a list of `torch.Tensor`'s.
        model: A `torch.nn.Module` or a list of `torch.nn.Module`.
        forward_only:
        tensor_shape: Shape of tensor. Required for P2P communication.
    """

    def __post_init__(self) -> None:
        self.validate_model()
        self.losses_reduced: List[torch.Tensor] = []
        # Input, output tensors only need to be saved when doing backward passes
        self.input_tensors = None
        self.output_tensors = None
        if not self.forward_only:
            self.input_tensors = []
            self.output_tensors = []

    @abc.abstractmethod
    def _validate_model(self) -> None:
        ...

    @abc.abstractmethod
    def run_warmup(self) -> None:
        ...

    @abc.abstractmethod
    def run_steady_state(self) -> None:
        ...

    @abc.abstractmethod
    def run_cooldown(self) -> None:
        ...

    def __call__(self) -> List[torch.Tensor]:
        self.run_warmup()
        self.run_steady_state()
        self.run_cooldown()
        return self.losses_reduced
