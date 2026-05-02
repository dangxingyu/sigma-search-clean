def f(sigma, state):
    """Damp the current largest singular direction; leave the rest Muon-like."""
    import torch

    scale = torch.ones_like(sigma)
    if sigma.numel() == 0:
        return scale

    sigma_flat = sigma.detach().reshape(-1)
    scale_flat = scale.reshape(-1)
    scale_flat[torch.argmax(sigma_flat)] = 0.5
    return scale
