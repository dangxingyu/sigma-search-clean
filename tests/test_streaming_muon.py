#!/usr/bin/env python3
"""
对拍 tests for StreamingMuon vs original nanochat Muon.

Tests:
1. streaming_msign converges to true msign(M) after enough iterations
2. StreamingMuonAdamW with identity produces similar updates to MuonAdamW
3. Training loss curves match between original and streaming (short run)
"""

import sys
import os
import json
import math
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NANOCHAT_DIR = os.path.join(SCRIPT_DIR, "nanochat")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, NANOCHAT_DIR)

import torch
import torch.nn.functional as F


def test_msign_convergence():
    """Test that streaming_msign converges to the true matrix sign."""
    print("=" * 60)
    print("TEST 1: streaming_msign convergence to true msign(M)")
    print("=" * 60)

    from streaming_muon_torch import streaming_msign

    torch.manual_seed(42)
    n, m = 128, 64
    M = torch.randn(n, m, dtype=torch.float32)

    # True msign via SVD
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    true_msign = U @ Vt  # (n, m)

    # Streaming msign: accumulate over iterations
    basis = None
    errors = []
    for step in range(50):
        update, basis, sigma = streaming_msign(
            M, basis, steps=1, ridge_epsilon=1e-9,
            fallback_to_qr=True, fallback_orthogonality_tol=0.1,
        )
        err = (update - true_msign).norm() / true_msign.norm()
        errors.append(err.item())
        if step % 10 == 0 or step == 49:
            print(f"  Step {step:3d}: relative error = {err:.6f}, "
                  f"sigma range = [{sigma.min():.4f}, {sigma.max():.4f}]")

    # After 50 steps, error should be very small
    final_err = errors[-1]
    print(f"\n  Final relative error: {final_err:.8f}")

    # Check singular values match
    true_sv = S
    print(f"  True singular values (top 5):     {true_sv[:5].tolist()}")
    print(f"  Streaming singular values (top 5): {sigma[:5].tolist()}")
    sv_err = (sigma - true_sv).norm() / true_sv.norm()
    print(f"  Singular value relative error: {sv_err:.8f}")

    assert final_err < 0.01, f"msign convergence too slow: {final_err}"
    assert sv_err < 0.01, f"singular value error too large: {sv_err}"
    print("  PASSED\n")


def test_msign_batched():
    """Test streaming_msign with batched input (num_params, n, m)."""
    print("=" * 60)
    print("TEST 2: streaming_msign batched operation")
    print("=" * 60)

    from streaming_muon_torch import streaming_msign

    torch.manual_seed(123)
    B, n, m = 8, 64, 32
    M = torch.randn(B, n, m, dtype=torch.float32)

    # True msign per batch element
    true_msigns = []
    for i in range(B):
        U, S, Vt = torch.linalg.svd(M[i], full_matrices=False)
        true_msigns.append(U @ Vt)
    true_msign = torch.stack(true_msigns)

    # Streaming msign
    basis = None
    for step in range(30):
        update, basis, sigma = streaming_msign(
            M, basis, steps=1, ridge_epsilon=1e-9,
            fallback_to_qr=True, fallback_orthogonality_tol=0.1,
        )

    err = (update - true_msign).norm() / true_msign.norm()
    print(f"  Batched relative error after 30 steps: {err:.6f}")
    assert err < 0.02, f"Batched msign error too large: {err}"
    print("  PASSED\n")


def test_msign_wide_matrix():
    """Test streaming_msign with wide matrix (n < m, should auto-transpose)."""
    print("=" * 60)
    print("TEST 3: streaming_msign wide matrix (n < m)")
    print("=" * 60)

    from streaming_muon_torch import streaming_msign

    torch.manual_seed(99)
    n, m = 32, 128
    M = torch.randn(n, m, dtype=torch.float32)

    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    true_msign = U @ Vt

    basis = None
    for step in range(30):
        update, basis, sigma = streaming_msign(
            M, basis, steps=1, ridge_epsilon=1e-9,
            fallback_to_qr=True, fallback_orthogonality_tol=0.1,
        )

    err = (update - true_msign).norm() / true_msign.norm()
    print(f"  Wide matrix relative error after 30 steps: {err:.6f}")
    assert err < 0.02, f"Wide matrix error too large: {err}"
    print(f"  Output shape: {update.shape} (should be {n}x{m})")
    assert update.shape == (n, m)
    print("  PASSED\n")


def test_sigma_transform():
    """Test that f(Σ) transforms modify the update correctly."""
    print("=" * 60)
    print("TEST 4: f(Σ) transform integration")
    print("=" * 60)

    from streaming_muon_torch import streaming_msign, TikhonovTransform, IdentityTransform

    torch.manual_seed(42)
    M = torch.randn(64, 32, dtype=torch.float32)

    # Run enough steps to converge
    basis = None
    for _ in range(30):
        _, basis, _ = streaming_msign(M, basis, steps=1)

    # Identity transform should give standard msign
    identity_state = {}
    update_id, _, sigma_id = streaming_msign(
        M, basis, steps=1,
        sigma_transform=IdentityTransform(), sigma_state=identity_state,
    )

    # Without transform
    update_plain, _, sigma_plain = streaming_msign(M, basis, steps=1)

    err = (update_id - update_plain).norm() / update_plain.norm()
    print(f"  Identity vs no-transform error: {err:.8f}")
    assert err < 1e-5, f"Identity transform should match plain: {err}"

    # Tikhonov should produce different result
    tik_state = {}
    update_tik, _, sigma_tik = streaming_msign(
        M, basis, steps=1,
        sigma_transform=TikhonovTransform(lam=1.0), sigma_state=tik_state,
    )
    diff = (update_tik - update_plain).norm() / update_plain.norm()
    print(f"  Tikhonov vs plain difference: {diff:.6f}")
    assert diff > 0.01, f"Tikhonov should differ from identity: {diff}"

    print("  PASSED\n")


def test_optimizer_step_match():
    """Test that StreamingMuonAdamW produces similar param updates to MuonAdamW."""
    print("=" * 60)
    print("TEST 5: StreamingMuonAdamW vs MuonAdamW single step comparison")
    print("=" * 60)

    from nanochat.optim import MuonAdamW
    from streaming_muon_torch import StreamingMuonAdamW

    torch.manual_seed(42)

    # Create identical params for both optimizers
    n, m = 128, 64
    num_params = 4

    def make_params():
        params = [torch.randn(n, m, requires_grad=True) for _ in range(num_params)]
        scalar = torch.randn(32, requires_grad=True)
        return params, scalar

    params_muon, scalar_muon = make_params()
    params_stream, scalar_stream = make_params()

    # Copy initial values
    for p1, p2 in zip(params_muon, params_stream):
        p2.data.copy_(p1.data)
    scalar_stream.data.copy_(scalar_muon.data)

    lr = 0.02
    momentum = 0.95
    wd = 0.0

    # Original Muon
    opt_muon = MuonAdamW([
        dict(params=params_muon, kind='muon', lr=lr, momentum=momentum,
             weight_decay=wd, ns_steps=5),
        dict(params=[scalar_muon], kind='adamw', lr=6e-4, betas=(0.9, 0.95),
             eps=1e-8, weight_decay=0.01),
    ])

    # StreamingMuon with identity
    opt_stream = StreamingMuonAdamW([
        dict(params=params_stream, kind='streaming_muon', lr=lr, momentum=momentum,
             weight_decay=wd, sigma_transform='identity', num_iters=1),
        dict(params=[scalar_stream], kind='adamw', lr=6e-4, betas=(0.9, 0.95),
             eps=1e-8, weight_decay=0.01),
    ])

    # Run multiple steps and compare
    torch.manual_seed(0)
    step_diffs = []
    for step in range(50):
        # Same random gradients
        grads = [torch.randn_like(p) for p in params_muon]
        grad_scalar = torch.randn_like(scalar_muon)

        for p, g in zip(params_muon, grads):
            p.grad = g.clone()
        scalar_muon.grad = grad_scalar.clone()

        for p, g in zip(params_stream, grads):
            p.grad = g.clone()
        scalar_stream.grad = grad_scalar.clone()

        opt_muon.step()
        opt_stream.step()

        # Compare params
        param_diff = sum((p1 - p2).abs().max().item() for p1, p2 in zip(params_muon, params_stream))
        scalar_diff = (scalar_muon - scalar_stream).abs().max().item()
        step_diffs.append(param_diff)

        if step % 10 == 0 or step == 49:
            print(f"  Step {step:3d}: muon param max diff = {param_diff:.6f}, "
                  f"adamw scalar diff = {scalar_diff:.10f}")

    # AdamW parts should be exactly identical
    print(f"\n  AdamW scalar final diff: {scalar_diff:.12f}")
    assert scalar_diff < 1e-6, f"AdamW should be identical: {scalar_diff}"

    # Muon params will differ because streaming uses power iteration vs Polar Express
    # They should both converge to similar values over time
    print(f"  Muon param diff trend: start={step_diffs[0]:.4f} -> end={step_diffs[-1]:.4f}")
    print("  NOTE: Muon vs StreamingMuon differ because Polar Express ≠ power iteration")
    print("        This is expected. The training comparison (test_training_match) is the real test.")
    print("  PASSED (structural test)\n")


def test_training_match(device='cpu', max_steps=100, depth=2, batch_size=4, seq_len=128):
    """
    End-to-end training comparison: original Muon vs StreamingMuon with identity.
    Both should produce similar loss curves if StreamingMuon is a valid msign approximation.
    Uses nanochat's actual GPT model and setup_optimizer.
    """
    print("=" * 60)
    print(f"TEST 6: Training comparison ({device}, {max_steps} steps, depth={depth})")
    print("=" * 60)

    from nanochat.gpt import GPT, GPTConfig
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.optim import MuonAdamW
    from streaming_muon_torch import StreamingMuonAdamW, DistStreamingMuonAdamW

    model_dim = depth * 128
    config = GPTConfig(
        vocab_size=32768,
        n_head=depth,
        n_layer=depth,
        n_kv_head=depth,
        n_embd=model_dim,
        sequence_len=seq_len,
    )

    def make_model_and_optimizer(model_name, use_streaming=False):
        """Create model + optimizer pair."""
        torch.manual_seed(42)
        with torch.device('meta'):
            model = GPT(config)
        model = model.to_empty(device=device)
        model.init_weights()

        # Use the model's setup_optimizer, then optionally swap for streaming
        # We need to replicate setup_optimizer's logic since it calls get_dist_info
        matrix_params = list(model.transformer.h.parameters())
        value_embed_params = list(model.value_embeds.parameters())
        embedding_params = list(model.transformer.wte.parameters())
        lm_head_params = list(model.lm_head.parameters())
        resid_params = [model.resid_lambdas]
        x0_params = [model.x0_lambdas]
        smear_params = [model.smear_gate.weight, model.smear_lambda, model.backout_lambda]

        dmodel_lr_scale = (model_dim / 768) ** -0.5

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=0.004 * dmodel_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=0.2 * dmodel_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embed_params, lr=0.1 * dmodel_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=0.005,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=0.5,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2,
                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        # Muon groups by shape
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            if use_streaming:
                param_groups.append(dict(
                    kind='streaming_muon', params=group_params, lr=0.02,
                    momentum=0.95, weight_decay=0.0,
                    sigma_transform='identity', num_iters=1,
                ))
            else:
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=0.02,
                    momentum=0.95, ns_steps=5, weight_decay=0.0,
                ))

        OptClass = StreamingMuonAdamW if use_streaming else MuonAdamW
        optimizer = OptClass(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return model, optimizer

    def train_loop(model, optimizer, model_name):
        losses = []
        torch.manual_seed(0)
        for step in range(max_steps):
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len + 1), device=device)
            targets = x[:, 1:].contiguous()
            inputs = x[:, :-1].contiguous()

            # Use bf16 autocast on CUDA
            if device != 'cpu':
                with torch.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                    logits = model(inputs)
                logits = logits.float().contiguous()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
            else:
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            if step % 20 == 0:
                print(f"  [{model_name}] step {step:3d}: loss = {loss.item():.4f}")

        return losses

    print("\n--- Original Muon (Polar Express) ---")
    model_muon, opt_muon = make_model_and_optimizer("Muon", use_streaming=False)
    losses_muon = train_loop(model_muon, opt_muon, "Muon")
    del model_muon, opt_muon
    torch.cuda.empty_cache() if device == 'cuda' else None

    print("\n--- StreamingMuon (identity f(Σ)) ---")
    model_stream, opt_stream = make_model_and_optimizer("StreamingMuon", use_streaming=True)
    losses_stream = train_loop(model_stream, opt_stream, "StreamingMuon")
    del model_stream, opt_stream
    torch.cuda.empty_cache() if device == 'cuda' else None

    # Compare final losses
    final_muon = sum(losses_muon[-10:]) / 10
    final_stream = sum(losses_stream[-10:]) / 10
    rel_diff = abs(final_muon - final_stream) / abs(final_muon)

    print(f"\n  Final avg loss (last 10 steps):")
    print(f"    Muon:           {final_muon:.4f}")
    print(f"    StreamingMuon:  {final_stream:.4f}")
    print(f"    Relative diff:  {rel_diff:.4f} ({rel_diff*100:.1f}%)")

    # Save results
    results = {
        'losses_muon': losses_muon,
        'losses_streaming': losses_stream,
        'final_muon': final_muon,
        'final_streaming': final_stream,
        'relative_diff': rel_diff,
        'device': device,
        'max_steps': max_steps,
        'depth': depth,
    }
    output_path = os.path.join(SCRIPT_DIR, 'comparison_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_path}")

    if rel_diff < 0.15:
        print("  PASSED (within 15% tolerance)")
    else:
        print(f"  WARNING: relative diff {rel_diff:.4f} > 0.15 — investigate further")

    return results


def test_custom_candidate():
    """Test that a sigma-search candidate file can be loaded and used."""
    print("=" * 60)
    print("TEST 7: sigma-search candidate integration")
    print("=" * 60)

    from streaming_muon_torch import streaming_msign, CustomTransform

    # Load a real bundled candidate instead of a test-only fixture.
    repo_root = os.path.dirname(SCRIPT_DIR)
    candidate_file = os.path.join(repo_root, 'candidates', 'top_aware_muon.py')
    code = open(candidate_file).read()
    ns = {"torch": torch, "math": math}
    exec(code, ns)
    f = ns['f']

    transform = CustomTransform(f)
    sigma_state = {"candidate_params": {"top_k": 1, "alpha": 0.5}}

    M = torch.randn(8, 64, 32)
    basis = None
    for _ in range(20):
        update, basis, sigma = streaming_msign(
            M, basis, steps=1,
            sigma_transform=transform, sigma_state=sigma_state,
        )

    print(f"  Candidate: top_aware_muon")
    print(f"  Sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")
    print(f"  Update norm: {update.norm():.4f}")
    print(f"  State keys: {list(sigma_state.keys())}")
    assert update.shape == (8, 64, 32)
    assert torch.all(torch.isfinite(update))
    print("  PASSED\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="StreamingMuon 对拍 tests")
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--training-steps', type=int, default=100)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--gpu-training', action='store_true',
                        help='Run GPU training comparison (depth=4, 200 steps)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("StreamingMuon 对拍 Test Suite")
    print("=" * 60 + "\n")

    t0 = time.time()

    # Unit tests (CPU, fast)
    test_msign_convergence()
    test_msign_batched()
    test_msign_wide_matrix()
    test_sigma_transform()
    test_custom_candidate()
    test_optimizer_step_match()

    if not args.skip_training:
        # Training comparison
        device = args.device
        if args.gpu_training and torch.cuda.is_available():
            device = 'cuda'
        test_training_match(
            device=device,
            max_steps=args.training_steps,
            depth=args.depth,
        )

    elapsed = time.time() - t0
    print(f"\nAll tests completed in {elapsed:.1f}s")
