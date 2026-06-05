from __future__ import annotations

import torch


class Optim:
    def __init__(self, params, method: str, lr: float, clip: float | None, lr_decay: float = 1.0, weight_decay: float = 0.0):
        self.params = list(params)
        self.method = method.lower()
        self.lr = lr
        self.clip = clip
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.optimizer = self._make_optimizer()

    def _make_optimizer(self):
        if self.method == "sgd":
            return torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay)
        if self.method == "adagrad":
            return torch.optim.Adagrad(self.params, lr=self.lr, weight_decay=self.weight_decay)
        if self.method == "adadelta":
            return torch.optim.Adadelta(self.params, lr=self.lr, weight_decay=self.weight_decay)
        if self.method == "adam":
            return torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Invalid optimizer: {self.method}")

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        grad_norm = None
        if self.clip is not None and self.clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        return grad_norm
