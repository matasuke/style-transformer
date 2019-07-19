from typing import List, Optional

import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torch.nn.utils import clip_grad_norm_

OPT_LIST = ["sgd", "adagrad", "adadelta", "adam"]
SCHEDULER_LIST = ["lambda", "step", "multistep", "exponential", "cosineannealing"]


OPTIMIZERS = {
    "sgd": optim.SGD,
    "adagrad": optim.Adagrad,
    "adadelta": optim.Adadelta,
    "adam": optim.Adam,
}

SCHEDULERS = {
    "lambda": scheduler.LambdaLR,
    "step": scheduler.StepLR,
    "multistep": scheduler.MultiStepLR,
    "exponential": scheduler.ExponentialLR,
    "cosineannealing": scheduler.CosineAnnealingLR,
}


class Optim:
    """
    wrapper for optimizer
    """
    __slot__ = [
        "last_metric",
        "lr",
        "max_grad_norm",
        "method",
        "scheduler",
        "weight_decay",
        "lr_decay",
        "start_decay_at",
        "start_decay"
    ]

    def __init__(
        self,
        method: str,
        lr: float,
        max_grad_norm: Optional[float] = None,
        lr_scheduler: Optional[str] = None,
        weight_decay: int = 0,
        lr_decay: int = 1,
        start_decay_at: Optional[int] = None,
    ) -> None:
        """
        wrap optimizer

        :param method: method to optimize. choose from sgd, adagrad, adadelta, adam.
        :param lr: learning rate
        :param max_grad_norm: maximum gradient norm
        :param lr_scheduler: learning rate scheduler method,
            choose from lambda, step, multistep, exponential, cosineannealing
            see more detail at "https://pytorch.org/docs/stable/optim.html"
        :param weight_decay: L2 penalty for optimizer
        :param lr_decay: decay ratio for learning rate
        :param start_decay_at: start learning rate decay at this epoch
        """
        msg = f"Unknown optimizer method: {method}"
        assert method in OPT_LIST, msg
        msg = f"Unknown scheduler method: {scheduler}"
        assert lr_scheduler is None or lr_scheduler in SCHEDULER_LIST, msg
        msg = f"one of lr_scheduler or start_decay_at must be None"
        assert lr_scheduler is None or start_decay_at is None, msg

        self.last_metric = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

    def set_parameters(self, params: List, **kwargs):
        """
        set parameters to optimizer
        parameters has to be list.

        :param params: parameters to be updated
        """
        self.params = params
        self.optimizer = OPTIMIZERS[self.method](
            self.params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.scheduler is not None:
            self.optimizer = SCHEDULERS[self.scheduler](self.optimizer, **kwargs)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def update_learning_rate(self, metric: float, step: int):
        """
        decay learning rate if val pref does not improve or we hit the start_decay_at limit

        :param ppl: last perplexity
        :param step: current step
        """
        if self.start_decay_at is not None and step >= self.start_decay_at:
            self.start_decay = True
        if self.last_metric is not None and metric > self.last_metric:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print(f"Decaying learning rate to {self.lr}")

        self.last_metric = metric
        self.optimizer.param_groups[0]["lr"] = self.lr
