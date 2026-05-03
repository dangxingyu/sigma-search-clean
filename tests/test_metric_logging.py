from __future__ import annotations

import torch

from metric_logging import hessian_power_probe


class _MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = torch.nn.Linear(1, 1, bias=False)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP()


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = torch.nn.ModuleList([_Block()])


class _ScalarQuadraticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Transformer()
        with torch.no_grad():
            self.transformer.h[0].mlp.c_fc.weight.fill_(1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        pred = self.transformer.h[0].mlp.c_fc(x)
        return 0.5 * torch.mean(pred * pred)


def test_hessian_probe_accumulates_microbatches_by_token_count() -> None:
    model = _ScalarQuadraticModel()
    name = "transformer.h.0.mlp.c_fc.weight"

    first = (torch.tensor([[1.0], [1.0]]), torch.zeros(2, 1))
    second = (torch.tensor([[3.0]]), torch.zeros(1, 1))
    stats = hessian_power_probe(
        model,
        [first, second],
        references={},
        name_to_param=dict(model.named_parameters()),
        top_k=1,
        iters=2,
        module_regex=None,
        selected_names=[name],
        return_stats=False,
    )

    # Token-weighted Hessian is (1^2 + 1^2 + 3^2) / 3 = 11 / 3.
    assert abs(stats["global_sharpness"] - (11.0 / 3.0)) < 1e-5
    assert stats["metadata"]["local_hessian_batches"] == 2
    assert stats["metadata"]["local_hessian_tokens"] == 3
    assert stats["metadata"]["global_hessian_tokens"] == 3
