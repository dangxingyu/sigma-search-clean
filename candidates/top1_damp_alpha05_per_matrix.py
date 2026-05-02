def f(sigma, state):
    """Damp the largest singular direction in each matrix by alpha=0.5."""
    import torch

    if sigma.numel() == 0:
        return torch.ones_like(sigma)

    scale = torch.ones_like(sigma)
    top_idx = torch.argmax(sigma.detach(), dim=-1, keepdim=True)
    scale.scatter_(-1, top_idx, 0.5)
    return scale
