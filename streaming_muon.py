# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.scipy.linalg import solve_triangular
from optax import tree_utils as otu

import haliax as hax
from haliax.tree_util import scan_aware_tree_map

from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import _weight_decay_hyperparam
from levanter.optim.util import (
    flatten_linear_layers,
    label_linear_like_module,
    unflatten_linear_layers,
)
from levanter.utils.jax_utils import leaf_key_paths


def _matmul_high_precision(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return jnp.matmul(lhs, rhs, precision=lax.Precision.HIGHEST)


def _symmetrize(matrix: jax.Array) -> jax.Array:
    return 0.5 * (matrix + matrix.T)


def _right_solve_upper(matrix: jax.Array, upper: jax.Array) -> jax.Array:
    # SCQR requires Q = A R^{-1}; solving on the transposed system avoids forming R^{-1}.
    return solve_triangular(upper, matrix.T, trans="T", lower=False).T


def _qr_with_positive_diagonal(matrix: jax.Array) -> jax.Array:
    q, r = jnp.linalg.qr(matrix, mode="reduced")
    diag_sign = jnp.sign(jnp.diag(r))
    diag_sign = jnp.where(diag_sign == 0, 1.0, diag_sign)
    return q * diag_sign[None, :]


def _diagonal_regularizer_from_gram(gram: jax.Array, ridge_epsilon: float) -> jax.Array:
    diagonal = jnp.maximum(jnp.diag(gram), 0.0)
    ridge_floor = jnp.finfo(gram.dtype).eps
    ridge_diagonal = jnp.maximum(diagonal * ridge_epsilon, ridge_floor)
    return jnp.diag(ridge_diagonal)


def _scalar_regularizer_from_gram(gram: jax.Array, ridge_epsilon: float) -> jax.Array:
    scale = jnp.linalg.norm(gram, ord="fro")
    ridge = jnp.maximum(scale * ridge_epsilon, jnp.finfo(gram.dtype).eps)
    return ridge * jnp.eye(gram.shape[0], dtype=gram.dtype)


def _orthogonality_error(matrix: jax.Array) -> jax.Array:
    gram = _matmul_high_precision(matrix.T, matrix)
    identity = jnp.eye(gram.shape[0], dtype=gram.dtype)
    return jnp.linalg.norm(gram - identity, ord="fro") / gram.shape[0]


def _orthogonalize_columns(
    matrix: jax.Array,
    *,
    gram: jax.Array | None,
    ridge_epsilon: float,
    fallback_to_qr: bool,
    diagonal_regularization: bool = True,
    fallback_orthogonality_tol: float | None = None,
) -> jax.Array:
    if gram is None:
        gram = _matmul_high_precision(matrix.T, matrix)

    gram = _symmetrize(gram)
    if diagonal_regularization:
        regularized = gram + _diagonal_regularizer_from_gram(gram, ridge_epsilon)
    else:
        regularized = gram + _scalar_regularizer_from_gram(gram, ridge_epsilon)
    chol_upper = jnp.linalg.cholesky(regularized).T
    scqr_q = _right_solve_upper(matrix, chol_upper)

    if not fallback_to_qr:
        return scqr_q

    scqr_failed = jnp.logical_or(jnp.any(~jnp.isfinite(chol_upper)), jnp.any(~jnp.isfinite(scqr_q)))
    if fallback_orthogonality_tol is not None:
        scqr_failed = jnp.logical_or(scqr_failed, _orthogonality_error(scqr_q) > fallback_orthogonality_tol)
    return lax.cond(
        scqr_failed,
        lambda _: _qr_with_positive_diagonal(matrix),
        lambda _: scqr_q,
        operand=None,
    )


def _column_normalize(matrix: jax.Array, eps: float) -> jax.Array:
    norms = jnp.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / jnp.maximum(norms, eps)


def _streaming_msign(
    matrix: jax.Array,
    previous_basis: jax.Array | None,
    *,
    steps: int,
    muon_eps: float,
    ridge_epsilon: float,
    fallback_to_qr: bool,
    diagonal_regularization: bool = True,
    fallback_orthogonality_tol: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    working = matrix.astype(jnp.float32)
    transposed = working.shape[0] < working.shape[1]
    if transposed:
        working = working.T

    basis_dim = working.shape[1]
    if previous_basis is None or previous_basis.shape != (basis_dim, basis_dim):
        basis = jnp.eye(basis_dim, dtype=jnp.float32)
    else:
        basis = previous_basis.astype(jnp.float32)

    for _ in range(steps):
        gram_matrix = _matmul_high_precision(working.T, working)
        first_pass = _matmul_high_precision(gram_matrix, basis)
        first_gram = _matmul_high_precision(basis.T, first_pass)
        second_pass = _orthogonalize_columns(
            first_pass,
            gram=first_gram,
            ridge_epsilon=ridge_epsilon,
            fallback_to_qr=fallback_to_qr,
            diagonal_regularization=diagonal_regularization,
            fallback_orthogonality_tol=None,
        )
        basis = _orthogonalize_columns(
            second_pass,
            gram=None,
            ridge_epsilon=ridge_epsilon,
            fallback_to_qr=fallback_to_qr,
            diagonal_regularization=diagonal_regularization,
            fallback_orthogonality_tol=fallback_orthogonality_tol,
        )

    rotated = _matmul_high_precision(working, basis)
    left_vectors = _column_normalize(rotated, muon_eps)
    update = _matmul_high_precision(left_vectors, basis.T)

    if transposed:
        update = update.T

    return update.astype(matrix.dtype), basis


class ScaleByStreamingMuonState(NamedTuple):
    """State for Muon with streaming power-iteration bases."""

    momentum_buffer: optax.Updates
    basis_cache: optax.Updates


class _StreamingMuonLeafResult(NamedTuple):
    update: object
    basis: object


def _init_basis_cache(params):
    flattened_params = flatten_linear_layers(params)

    def init_leaf(param):
        if isinstance(param, hax.nn.Linear) and param.weight is not None and param.weight.array is not None:
            basis_dim = min(param.weight.array.shape)
            return jnp.eye(basis_dim, dtype=jnp.float32)
        return None

    return scan_aware_tree_map(
        init_leaf, flattened_params, is_leaf=lambda x: isinstance(x, hax.nn.Linear) or x is None
    )


def scale_with_streaming_muon(
    momentum=0.95,
    nesterov=True,
    steps=1,
    muon_eps=1e-8,
    use_kimi_scaling=False,
    ridge_epsilon=1e-9,
    fallback_to_qr=True,
    diagonal_regularization=True,
    fallback_orthogonality_tol=None,
):
    steps = int(steps)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        basis_cache = _init_basis_cache(params)
        return ScaleByStreamingMuonState(momentum_buffer=momentum_buffer, basis_cache=basis_cache)

    def update_fn(updates, state, params=None):
        del params

        grad_coeff = 1.0 - momentum
        momentum_buffer = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + grad_coeff * g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )

        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + grad_coeff * g,
                momentum_buffer,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = momentum_buffer

        flattened_updates = flatten_linear_layers(updates)

        def transform_leaf(update, previous_basis):
            if isinstance(update, hax.nn.Linear) and update.weight is not None and update.weight.array is not None:
                transformed_weight_array, new_basis = _streaming_msign(
                    update.weight.array,
                    previous_basis,
                    steps=steps,
                    muon_eps=muon_eps,
                    ridge_epsilon=ridge_epsilon,
                    fallback_to_qr=fallback_to_qr,
                    diagonal_regularization=diagonal_regularization,
                    fallback_orthogonality_tol=fallback_orthogonality_tol,
                )

                if not use_kimi_scaling:
                    scale = jnp.sqrt(
                        jnp.maximum(1.0, transformed_weight_array.shape[0] / transformed_weight_array.shape[1])
                    )
                else:
                    scale = 0.2 * jnp.sqrt(
                        jnp.maximum(transformed_weight_array.shape[0], transformed_weight_array.shape[1])
                    )

                transformed_weight = dataclasses.replace(update.weight, array=transformed_weight_array * scale)
                transformed_update = dataclasses.replace(update, weight=transformed_weight)
                return _StreamingMuonLeafResult(update=transformed_update, basis=new_basis)

            return _StreamingMuonLeafResult(update=update, basis=previous_basis)

        transformed = scan_aware_tree_map(
            transform_leaf,
            flattened_updates,
            state.basis_cache,
            is_leaf=lambda x: isinstance(x, hax.nn.Linear) or x is None,
        )
        transformed_updates = jax.tree.map(
            lambda leaf: leaf.update,
            transformed,
            is_leaf=lambda x: isinstance(x, _StreamingMuonLeafResult),
        )
        new_basis_cache = jax.tree.map(
            lambda leaf: leaf.basis,
            transformed,
            is_leaf=lambda x: isinstance(x, _StreamingMuonLeafResult),
        )
        updates = unflatten_linear_layers(updates, transformed_updates)

        return updates, ScaleByStreamingMuonState(momentum_buffer=momentum_buffer, basis_cache=new_basis_cache)

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("streaming_muon")
@dataclass(frozen=True)
class StreamingMuonConfig(OptimizerConfig):
    """
    Muon optimizer using streaming power iteration with warm-started bases and
    shifted Cholesky QR fallback instead of Newton-Schulz orthogonalization.
    """

    learning_rate: float = 0.02
    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 1
    weight_decay: float = 0.0
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    adamc_weight_decay: bool = False
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False
    ridge_epsilon: float = 1e-9
    fallback_to_qr: bool = True
    diagonal_regularization: bool = True
    fallback_orthogonality_tol: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        weight_decay_hyperparam = _weight_decay_hyperparam(
            self.weight_decay,
            learning_rate_schedule=learning_rate_schedule,
            peak_learning_rate=self.learning_rate,
            adamc_weight_decay=self.adamc_weight_decay,
        )
        adam_base_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
        adam_weight_decay_hyperparam = _weight_decay_hyperparam(
            adam_base_weight_decay,
            learning_rate_schedule=adam_lr_schedule,
            peak_learning_rate=self.adam_lr,
            adamc_weight_decay=self.adamc_weight_decay,
        )

        def optimizer(learning_rate, adam_lr, weight_decay, adam_weight_decay):
            def muon_transform():
                components = []
                components.append(
                    scale_with_streaming_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                        self.ridge_epsilon,
                        self.fallback_to_qr,
                        self.diagonal_regularization,
                        self.fallback_orthogonality_tol,
                    )
                )
                components.append(optax.add_decayed_weights(weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            weight_decay=weight_decay_hyperparam,
            adam_weight_decay=adam_weight_decay_hyperparam,
        )

    def create_mask(self, params, use_kimi_scaling=True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            if isinstance(param, hax.nn.Linear):
                assert param._out_first or use_kimi_scaling
                return label_linear_like_module(param, weight_label="muon", bias_label="adamw")
            return "adamw"

        return hax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, hax.nn.Linear))
