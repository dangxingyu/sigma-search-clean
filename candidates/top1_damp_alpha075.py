"""Top-1 singular direction damping with alpha=0.75."""

import torch


def f(sigma, state):
    scale = torch.ones_like(sigma)
    if sigma.numel() == 0:
        return scale
    sigma_flat = sigma.detach().reshape(-1)
    scale_flat = scale.reshape(-1)
    scale_flat[torch.argmax(sigma_flat)] = 0.75
    return scale
