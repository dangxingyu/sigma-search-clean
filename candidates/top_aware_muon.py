"""Top-Aware Muon candidate.

Scaling rule: damp the per-matrix top-k singular directions by alpha and leave
all other directions Muon-like. Configure from run_eval.py with:

    --candidate-param top_k=1 --candidate-param alpha=0.5
"""

CANDIDATE_NAME = "Top-Aware Muon"


def _param(state, name, default):
    params = state.get("candidate_params", {})
    if name in params:
        return params[name]
    if name in state:
        return state[name]
    return default


def f(sigma, state):
    import torch

    if sigma.numel() == 0:
        return torch.ones_like(sigma)

    top_k = int(_param(state, "top_k", 1))
    alpha = float(_param(state, "alpha", 0.5))
    k = min(max(top_k, 0), sigma.shape[-1])

    scale = torch.ones_like(sigma)
    if k == 0:
        return scale

    top_idx = torch.topk(sigma.detach(), k=k, dim=-1).indices
    scale.scatter_(-1, top_idx, alpha)
    return scale
