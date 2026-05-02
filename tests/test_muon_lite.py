"""Quick smoke test for muon_lite.py — verifies shapes, chi=1 equivalence,
and that chi=2 produces a different update."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "nanochat"))

import torch
from muon_lite import _polar_factor, _lite_top_eigvecs, _apply_lite, MuonLiteAdamW

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Test 1: shapes
g = torch.randn(3, 256, 128, device=device, dtype=torch.bfloat16)  # B=3, m=256, n=128 (tall)
X = _polar_factor(g)
assert X.shape == g.shape, f"X shape {X.shape} != {g.shape}"
print(f"polar factor shape OK: {X.shape}, dtype={X.dtype}")

V_s = _lite_top_eigvecs(g, d_s=4)
assert V_s.shape == (3, 128, 4), f"V_s shape {V_s.shape}"
print(f"V_s (tall) shape OK: {V_s.shape}")

# chi=1 should be identity op
X_chi1 = _apply_lite(X.float(), V_s, chi=1.0)
assert torch.allclose(X_chi1, X.float(), atol=1e-5), "chi=1 should be no-op"
print(f"chi=1 identity OK")

# chi=2 should modify
X_chi2 = _apply_lite(X.float(), V_s, chi=2.0)
diff = (X_chi2 - X.float()).norm().item()
x_norm = X.float().norm().item()
ratio = diff / x_norm
print(f"chi=2 relative change: {ratio:.4f} (should be ~0.1 if d_s=4/128 is sharp)")
assert ratio > 0.01, "chi=2 should change X appreciably"

# Test wide matrix
g_wide = torch.randn(3, 64, 256, device=device, dtype=torch.bfloat16)
X_wide = _polar_factor(g_wide)
V_s_wide = _lite_top_eigvecs(g_wide, d_s=4)
assert V_s_wide.shape == (3, 64, 4), f"V_s wide shape {V_s_wide.shape}"
X_wide_lite = _apply_lite(X_wide.float(), V_s_wide, chi=2.0)
assert X_wide_lite.shape == X_wide.shape
print(f"wide matrix LITE OK: {X_wide.shape}")

# Test 2: optimizer end-to-end, one step (LITE-L, no β)
params = [torch.randn(128, 128, device=device, requires_grad=True) for _ in range(4)]
for p in params: p.grad = torch.randn_like(p) * 0.01

pg = [dict(kind='muon_lite', params=params, lr=0.02, momentum=0.95,
           weight_decay=0.01, ns_steps=5, lite_chi=2.0, lite_ds=4,
           lite_beta1=0.0, lite_beta2=0.0)]
opt = MuonLiteAdamW(pg)
p0_before = params[0].detach().clone()
opt.step()
p0_after = params[0].detach()
delta_L = (p0_after - p0_before).norm().item()
print(f"LITE-L (χ only) step OK, |Δw|={delta_L:.4e}")
assert delta_L > 0 and torch.all(torch.isfinite(p0_after))

# Test 3: full LITE with β terms
params2 = [torch.randn(128, 128, device=device, requires_grad=True) for _ in range(4)]
for p in params2: p.grad = torch.randn_like(p) * 0.01

pg2 = [dict(kind='muon_lite', params=params2, lr=0.02, momentum=0.95,
            weight_decay=0.01, ns_steps=5, lite_chi=2.0, lite_ds=4,
            lite_beta1=-0.25, lite_beta2=1.0)]
opt2 = MuonLiteAdamW(pg2)
p0_before2 = params2[0].detach().clone()
opt2.step()
p0_after2 = params2[0].detach()
delta_full = (p0_after2 - p0_before2).norm().item()
print(f"full LITE (χ+β) step OK, |Δw|={delta_full:.4e}")
assert delta_full > 0 and torch.all(torch.isfinite(p0_after2))
# β term should change step magnitude noticeably
print(f"  full-vs-L step magnitude ratio: {delta_full/delta_L:.3f}")
print("All smoke tests passed.")
