"""Muon optimizer — Newton-Schulz orthogonalization for matrix gradients.

Standalone implementation with no external dependencies beyond PyTorch.
Reference: https://kellerjordan.github.io/posts/muon/

Muon applies Newton-Schulz orthogonalization to matrix-shaped gradients,
effectively normalizing the update direction. Best used for 2D weight matrices;
use a standard optimizer (AdamW) for biases, norms, and embeddings.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    """Orthogonalize a 2D gradient matrix using Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalized updates.

    Args:
        params: Parameters to optimize (should be 2D weight matrices).
        lr: Learning rate.
        momentum: Momentum coefficient (default 0.95).
        backend_steps: Newton-Schulz iteration steps (default 5).
        nesterov: Use Nesterov momentum (default True).
        weight_decay: Decoupled weight decay (default 0).
    """
    def __init__(self, params, lr: float, momentum: float = 0.95,
                 backend_steps: int = 5, nesterov: bool = True,
                 weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            # decoupled weight decay
            if wd > 0:
                for p in params:
                    p.data.mul_(1 - lr * wd)

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss
