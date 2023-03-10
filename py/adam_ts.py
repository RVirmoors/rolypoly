import torch
from typing import List, Tuple

@torch.jit.script
class Adam(object):
    def __init__(self, params: List[torch.Tensor], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0):
        self.lr: float = lr
        self.betas: Tuple[float, float] = betas
        self.eps: float = eps
        self.weight_decay: float = weight_decay
        self.steps: int = 0
        self.m: List[torch.Tensor] = [torch.zeros_like(p) for p in params]
        self.v: List[torch.Tensor] = self.m.copy()

    def zero_grad(self):
        for m, v in zip(self.m, self.v):
            m.zero_()
            v.zero_()

    def step(self, params: List[torch.Tensor], grads: List[torch.Tensor]):
        self.steps += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            if self.weight_decay != 0:
                g = g + self.weight_decay * p
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (g ** 2)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.steps)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.steps)
            new_p = p.detach().clone()
            new_p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            p.data.copy_(new_p)