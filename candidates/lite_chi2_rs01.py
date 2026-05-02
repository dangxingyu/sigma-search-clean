def f(sigma, state):
    """Streaming LITE-like transform: sharp top 10% scale 1, flat tail scale 2."""
    import torch

    if sigma.numel() == 0:
        return torch.ones_like(sigma)

    scale = torch.full_like(sigma, 2.0)
    rank = sigma.shape[-1]
    d_s = max(1, int(round(0.1 * rank)))
    top_idx = torch.topk(sigma.detach(), k=d_s, dim=-1).indices
    scale.scatter_(-1, top_idx, 1.0)
    return scale
