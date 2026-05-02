# New Thoughts — Reading Sadhika's guidance

## Current conclusion ledger addendum (2026-05-02, v42/v43 clean StreamingMuon study)

### 1. Confident enough to treat as correct

- The active clean-repo research path should compare `streaming_identity` (`c=1`) against Top-Aware Muon `top_k=1, alpha=0.5` (`c=0.5`) first. Native Muon/LITE code remains as deprecated controls, not the default handoff workflow.
- The clean StreamingMuon metrics path should not do explicit SVD. For this phase, use cached streaming `sigma`/basis, Nesterov-corrected optimizer input metrics, and Hessian/projection diagnostics.
- The v42/v43 evidence is internally consistent at 4M: Top-Aware `c=0.5` clearly beats identity under the same driver/recipe, both without metrics and with dense metrics enabled.

### 2. Multiple observations; likely true but still needs careful confirmation

- The Top-Aware advantage is clearly positive at 4M and 8M under the d8 0.4B-token clean recipe. Current best deltas for `c=0.5 - identity` are `-0.0323` at 4M and `-0.0346` at 8M.
- 262K looks like a no-benefit regime for `c=0.5`: identity is slightly better after LR tuning.
- Dense no-SVD metrics are feasible at d8: 4M runs with metrics every step and Hessian every 24 steps produced usable JSONs around `6.8MB` per run.

### 3. Some observations suggest

- Top-Aware can improve BPB while the measured selected-subspace sharpness is higher than identity. The mechanism is therefore not simply "reduce all measured sharpness"; it may be allowing useful progress while controlling the top sigma direction's effective update.
- The transition region is currently not monotone in one seed: 1M has a tiny LR-closed `c=0.5` edge, 2M favors identity after LR closure, and 4M favors `c=0.5`. This could be seed noise, schedule interaction, or a real nonmonotone finite-horizon effect.
- v46 2M dynamics does not provide an obvious scalar explanation: identity wins despite similar late sharpness/projection-correlation aggregates. This suggests the mechanism is likely trajectory- or spectrum-shape-dependent, not captured by the current aggregate sharpness alone.
- The fixed-Hessian-subspace gradient projection cosine is strongly negative late in 4M runs, suggesting oscillatory behavior in the sharp subspace. This needs more interpretation before becoming a central claim.

### 4. Hypotheses

- Top-Aware `c=0.5` helps in high-batch regimes because the top sigma direction imposes an edge-of-stability-style global LR bound; damping that direction lets the remaining directions use a larger effective stable LR.
- The transition between identity and `c=0.5` may not be described by batch size alone under the current finite-token schedule. A more stable predictor may require dynamics metrics, more seeds, or a schedule-normalized statistic rather than only final BPB at one seed.

## Current conclusion ledger addendum (2026-05-02)

### 1. Confident enough to treat as correct

- The standalone repo should not expose historical one-off launchers as primary interfaces. The stable pass-by surface is `run_eval.py`, `run_top_aware_muon_sweep.py`, `run_native_muon_v9.py`, `run_lite_v9.py`, `scripts/run_d8_metrics_grid.sh`, and `scripts/build_sweep_catalog.py`.
- The handoff sweep surface should now be `scripts/run_handoff_sweep.sh` for new users and `run_top_aware_muon_sweep.py` for direct control. Adaptive LR boundary extension is implemented and should be left on for optimizer-quality sweeps.
- Split-batch alignment for the current study must be computed on optimizer input `M'`, not raw `M`. The clean StreamingMuon and native Muon metric paths now use `M' = (1 - beta)G + beta M_new`.
- In 8-GPU runs, split-momentum diagnostics should average split gradients across DDP ranks before updating the diagnostic momenta. The corrected clean path now records global-DDP split alignment; previous rank-local split logs are weaker diagnostics.
- For the requested d8 dynamics study, `262144` is the critical batch. The no-tuning grid should be `{262K,1M,4M}`, not `{262K,512K,1M,4M}`.
- The d8 dynamics budget should be about `0.4B` tokens. `402,653,184` is a practical exact value because it is divisible by `262K`, `1M`, and `4M`.

### 2. Multiple observations; likely true but still needs careful confirmation

- At 128K fixed `lr=0.01`, near-identity Top-Aware `alpha=1.15` beats StreamingMuon identity in paired seeds `{42,43,44}` by mean `0.000509 ± 0.000057` BPB. This is a small fixed-LR signal, not yet a full LR-swept optimizer-quality claim.

### 3. Some observations suggest

- The expensive part of dense logging is likely every-step SVD/statistics more than Hessian alone. The corrected 200-step smoke confirms Hessian entries can be produced, but matched ablations are still needed for exact overhead attribution.

### 4. Hypothesis

- Dense metrics at d8 may cost much more than no-metrics training when `metrics_every=1`, primarily from per-step SVD diagnostics. This should be measured from elapsed times in matched runs rather than assumed. The d8 metrics grid records elapsed time in each result JSON for that purpose.

## Current conclusion ledger (2026-04-30)

### 1. Confident enough to treat as correct

- Sadhika's core diagnostic target is reproducibility of singular directions across independent data samples: high split-half alignment means resolved structure, low alignment means noise/bulk.
- Fixed LR or one-off LR comparisons are not acceptable here. The observed Muon-vs-LITE gap is very sensitive to LR and schedule, so winner claims require matched LR sweeps.
- The v8 d=12 sweep is bug-confounded: original DDP drivers omitted the `world_size` divisor in `grad_accum`, so effective batch is 8× nominal and LR scaling is wrong. Do not use v8 for clean d=12 scaling conclusions.
- The current LITE implementation is not distributed. Any fair Muon-vs-LITE run must either use world=1 for both or implement a real distributed LITE optimizer.
- LITE's algorithm acts on `M_tilde`, the Nesterov-corrected momentum direction, not directly on the raw gradient. Therefore the most faithful SVD diagnostic should inspect split-half `M_tilde`.
- LITE's sharp/flat projection is side-dependent: tall matrices use right singular vectors (`V`), wide matrices use left singular vectors (`U`). Sadhika's original `u_A·u_B` rule is a good starting diagnostic, but a LITE-specific diagnostic should be side-aware.
- v9 is a clean d=8 / 1B-token / 3-seed follow-up for momentum-D1: 24/24 runs completed with world=1 for both optimizers, avoiding the v8 DDP and distributed-LITE confounds.
- In v9, the deployable signal is absolute D1 on a reference or Muon trajectory, not `D1(LITE) - D1(Muon)`. The latter is near-zero or wrong-signed while LITE still wins at every tested batch size.
- v10 shows that a simple time-only triangular χ schedule (`1 -> 2 -> 1`) is not an improvement over fixed χ=2. It underperforms fixed LITE at every tested batch size, while still beating Muon.
- v11 shows that fixed lower χ=1.5 is not an improvement over fixed χ=2 in the 512K/1M crossover regimes under the current LR choices. It still beats Muon, but loses to χ=2 on every tested seed.
- v12 shows the χ=1.5 loss is not explained by LR mistuning at 512K/1M. After adjacent LR checks, the best χ=1.5 rows remain worse than fixed χ=2.
- v13 provides a clean local 256K lower-edge point with v9-style `M_tilde` telemetry: Muon and fixed χ=2 LITE are effectively tied, Δ `-0.0001 ± 0.0005`.
- v14 shows the first-probe binary global χ gate is not a winning hybrid: it preserves fixed-LITE behavior at 512K/1M but makes 256K worse than both Muon and fixed LITE.
- The main research question is optimizer switching, not hyperparameter tuning of LITE. v15 can inform mechanism, but conclusions for Sadhika should separate "better LITE variant" from "does Muon/LITE switching help?"
- v16 is a clean true one-switch test, not a χ proxy: matrix groups share momentum state, Muon phase uses native Muon-style polar update with aspect-ratio LR scaling, and LITE phase uses current `χ=2`, `r_s=0.1`.
- Under the tested v16 rule (`M_tilde` c* fraction threshold `0.417`), optimizer switching does not beat the better pure optimizer at 256K, 512K, or 1M. Muon->LITE is worse than pure LITE at 512K/1M; LITE->Muon is worse than pure LITE at 512K and ties at 256K.
- v17 shows that from the exact same Muon step170 checkpoint at 1M seed42, switching matrix groups to LITE is locally better than continuing Muon by `+0.001185` BPB, but still worse than pure LITE. Therefore v16's failure is not because LITE is locally bad after Muon; it is because the branch cannot recover the full pure-LITE trajectory.
- The focused mainline package is now `mainline_switching_results.md` + `mainline_switching_results.png`. It should be the default source for Sadhika-facing results because it is organized by question rather than by experiment number.
- Across the clean pure-baseline table `{256K,512K,1M,4M,16M}`, Muon-trajectory momentum-D1 remains strongly correlated with LITE advantage: `corr(Muon D1 last, Muon-LITE Δ) = +0.968`; `corr(Muon D1 first, Muon-LITE Δ) = +0.865`.
- StreamingMuon identity is a warm-started streaming-SVD approximation to Muon's matrix-sign update on the Nesterov momentum object, with optional per-singular-direction scaling `f(sigma)`. It is useful infrastructure for spectral transforms, but it is not itself an optimizer-switching rule.
- For sigma-search, StreamingMuon identity is a valid Muon-like baseline only with strict SCQR fallback control. Existing d8/1B seed42 evidence at matrix LR `0.02` shows `fallback_orthogonality_tol=0.01` or `0.02` matches native Muon closely (`0.87486/0.87499` vs same-LR native Muon `0.87409`), while loose tolerances `0.05/0.1` drift badly (`~1.20` best BPB, final `~1.35`).
- StreamingMuon has a distributed optimizer path (`DistStreamingMuonAdamW`) that now has fresh 8-GPU DDP smoke coverage after the 2026-04-29 wrapper fixes: identity, strict tolerances, grad accumulation, pure QR, mutable-candidate state, and base_train wrapper checkpointing all completed on 8×B200.
- In a matched 8-GPU DDP d8/seq1024/global-batch-262K probe, StreamingMuon identity with `fallback_orthogonality_tol=0.01` matches native Muon closely at a 1024-step horizon: native final BPB `0.995475`, streaming final BPB `0.994029`, delta native-streaming `+0.001447`.
- In a corrected larger d12/seq1024/global-batch-262K/4096-step run, StreamingMuon identity `tol=0.01` matches native Muon essentially exactly over ~1.07B tokens: native final BPB `0.866625`, streaming final BPB `0.866802`, delta native-streaming `-0.000177`.
- The current sigma-transform focus is no longer hard Muon/LITE switching. The active question is whether manipulating `f(sigma)` can relax the sharp-direction edge-of-stability LR bound while leaving river/flat directions with a larger effective LR.
- v18 does not yet close the 128K small-batch question. Within the tested grid `{0.01,0.02,0.04}`, LITE and Muon are effectively tied, but both optimizers choose the lower grid edge `0.01`. This must be extended to `0.005` and possibly `0.0025` before using 128K as a clean crossover point.
- Update after v18b: 128K is now low-LR closed over `{0.005,0.01,0.02,0.04}`. Both Muon and LITE prefer `lr=0.01`; `lr=0.005` is clearly worse. The best-LR comparison is still an effective tie with a tiny LITE edge (`+0.00012` BPB).
- After LR sweep, there is no robust small-batch Muon-over-LITE claim in the clean 128K data. Muon wins the non-optimal `lr=0.005` row and one of three seeds at `lr=0.01`, while LITE wins two of three seeds at the tuned row. The 256K point is also only a tiny Muon/tie (`-0.0001 ± 0.0005`). Phrase the current small-batch conclusion as "tie / no meaningful LITE advantage", not "Muon clearly wins".
- LR-sweep plots are now consolidated in `optimizer_lr_sweeps.png` and `optimizer_lr_sweeps_top1_8m.png`; raw aggregate data is in `optimizer_lr_sweeps_summary.json`.
- v26 is the right next test for "does Muon win at even smaller batch?" because it avoids the raw DDP-vs-single comparison problem: `identity`, `lite_chi2_rs01`, and `top1pm_a05` all run through the same StreamingMuon DDP driver with pure QR and two iterations. Treat old/native single-GPU LITE numbers as separate controls, not the primary top1 comparison.
- Scaling rule: do not launch d12/~3B just because one small-batch run finishes. The ideal pattern needed for scale-up is large-batch `top1 > LITE-like > identity` and small-batch `identity >= LITE-like`, after LR boundary closure. If that pattern holds, a d12/3B run is justified as a robustness/model-scale check; otherwise the next action is targeted LR/seed closure.

### 2. Multiple observations; likely true but still needs careful confirmation

- At d=8 / 1B tokens, fixed-χ LITE tends to improve over Muon as batch size grows, with crossover around the few-hundred-K to 1M token/step regime after LR tuning.
- Raw gradient D1 has the right monotonic direction with batch size, but Sadhika's absolute rule `>50% of c_i > 0.7 => LITE` does not transfer cleanly from Adam-warmup probes to in-training Muon trajectories.
- A lower/data-dependent threshold such as the current `c*` tracks outcomes better than the hard 0.7 cutoff in the existing in-training data.
- D1 values drift during training; early gradients have clearer resolved spikes than later gradients. This supports time-dependent or adaptive LITE rather than a single static switch.
- v9 strengthens the batch-size ramp under a cleaner optimizer-input probe: paired Δ_bpb is `+0.0024` at 512K, `+0.0048` at 1M, `+0.0124` at 4M, and `+0.0253` at 16M.
- Absolute Muon-trajectory D1(`M_tilde`) likely predicts the magnitude of LITE's advantage: last-probe c* values increase from `0.354/0.375` at 512K/1M to `0.792` at 4M and `1.000` at 16M.
- For momentum-D1, the current `c*` threshold is probably over-permissive at high batch. The hard `c > 0.7` fraction keeps more useful dynamic range across 512K, 1M, 4M, and 16M.
- v10 suggests fixed χ=2 is already close to optimal at 512K/1M under current LR, because triangular χ is only slightly worse there (`-0.0003` and `-0.0006` BPB vs fixed), not better.
- v11 strengthens the same point: lowering χ to 1.5 at 512K/1M makes final BPB worse than χ=2 by about `0.0008-0.0011`, so the borderline regimes do not obviously want weaker unconditional amplification.
- v12 further strengthens it: 512K χ=1.5 improves slightly by moving LR from `0.01` to `0.02`, but still remains `+0.0004` BPB worse than χ=2; 1M's best χ=1.5 LR remains `0.02` and is `+0.0011` worse.
- The cleanest missing calibration point is 256K with the same local v9-style driver and `M_tilde` telemetry. Existing 256K evidence is useful, but mixing old/Modal telemetry into the gate threshold is weaker than a local re-run.
- v13 shows late hard D1(`M_tilde`, `c>0.7`) is not enough as a switch scalar near crossover: 256K and 512K both end at `0.125` despite 256K tie/slight Muon and 512K LITE win. First/mean `c*` separates them better but only weakly.
- Gate calibration across clean 256K + v9 says first-probe `c*` is the best simple separator tested so far: 256K/512K margin `+0.125` with threshold about `0.417`. Mean `c*` is weaker but usable; late hard `c>0.7` has zero margin.
- Existing switching evidence is negative but not yet final: v14 is a χ-on/off proxy, not a true optimizer switch. It suggests that a correct regime classifier does not automatically produce a better training trajectory.
- v16 upgrades the switching evidence from proxy to direct: correct-looking switches still do not improve final BPB. The strongest negative case is 512K, where Muon->LITE switches early and LITE->Muon switches later, but both underperform pure LITE.
- v15 suggests global larger sharp rank is not a robust low-batch repair. `r_s=0.2` slightly helps 256K but slightly hurts 512K, while `r_s=0.5` hurts 512K more. If adaptive rank matters, it likely needs per-layer/data-dependent rank, not one global larger `r_s`.
- v17 is now the right causal follow-up: same Muon checkpoint at step 170, then continue Muon vs switch matrix groups to LITE. The resume sanity check is good because both branches have identical step-170 BPB `1.149135`; final result is still pending.

### 3. Some observations suggest

- The momentum/Nesterov diagnostic may be more relevant than raw gradient D1 because it measures the matrix the optimizer actually orthogonalizes or amplifies.
- Side-aware alignment may change the story for rectangular MLP matrices. If left- and right-side D1 disagree, the side-aware metric should be favored for predicting fixed-χ LITE.
- The current probes use top-k=16, while LITE's main-matrix `d_s` is about 10% of min-dim (≈51 at d=8). If directions 17-51 behave differently from the top 16, the diagnostic may be over-optimistic. Need a top-k sweep or a top-`d_s` aggregate.
- Fixed-β EMA alignment can saturate near 1 and may be too easy; the right test is D1 on scheduled `M_tilde`, plus raw momentum-buffer D1 as a secondary diagnostic.
- The v9 smoke already shows scheduled `M_tilde` alignment can saturate at 1.0 in a short 1M-batch run. Full v9 will tell whether this is just smoke/horizon bias or a real failure mode of momentum-D1 as a switch scalar.
- The first full v9 probes reinforce the saturation concern: at 16M D1(`M_tilde`, c*) is 1.0 for all first-probe seeds/opts, and at 4M first Muon probes D1(`M_tilde`, c*) is also 1.0 even while raw D1(g) is only 0.125-0.250.
- If momentum-D1 is retained, `frac(c>0.7)` may be more informative than `frac(c>c*)` for `M_tilde`; early v9 shows `c*` saturating while `0.7` still distinguishes 4M from 16M.
- Momentum-D1 appears to preserve a time-and-batch gradient: 16M stays saturated through all 5 probes, while 4M Muon declines materially through training. This supports using the whole trajectory or late-probe value, not just the first momentum-D1 snapshot.
- v9 partial result at 16M: LITE still beats Muon by about +0.025 BPB across three seeds, but D1(`M_tilde`) is saturated for both optimizers. So the 16M win is consistent with high alignment, but the saturated scalar cannot explain differences within the high-batch regime.
- Early v9 1M Muon probes are substantially lower than 4M and 16M on D1(`M_tilde`), suggesting momentum-D1 has batch-size ordering even though its c* threshold is over-permissive at high batch.
- D1 drift is not only a raw-gradient phenomenon; v9 shows D1(`M_tilde`) also declines through training at 1M and 4M. This strengthens the case for an adaptive/hybrid schedule instead of one fixed switch decision.
- The side-aware correction appears important, not cosmetic: at 1M Muon last-probe, LITE-side D1(`M_tilde`, c*) is 0.375 while left-U D1 is 0.167. Sadhika's original U-only diagnostic may under-call LITE-relevant alignment for tall matrices.
- v9 4M paired data suggests the useful diagnostic is absolute resolvability, not optimizer-trajectory contrast: LITE wins by about +0.012 BPB, but D1(`M_tilde`) is almost the same on Muon and LITE trajectories at the last probe.
- Partial v9 paired data suggests a practical switcher should probe a reference/Muon trajectory and use absolute D1(`M_tilde`) level; `D1(LITE)-D1(Muon)` is not a deployable metric and empirically is near-zero or wrong-signed despite LITE wins.
- 512K and 1M Muon have very similar low last-probe D1(`M_tilde`) values in v9, matching the fact that both are near the crossover/borderline regime rather than clearly high-batch LITE regimes.
- Full v9 preserves this pattern after 24/24 runs: 512K/1M are low-D1, small-win regimes; 4M is mid/high-D1 with a clear win; 16M is saturated-D1 with the largest win but also few optimizer steps.
- Side-aware alignment is not a cosmetic implementation detail. In full v9, 512K Muon last-probe `M_tilde` c* is `0.354` on the LITE side but `0.062` on Sadhika's left-U side.
- Momentum-D1 saturation at 16M is not a failure of the story, but it limits within-high-batch resolution. It can say "strong LITE regime" but cannot rank 16M variants or explain optimizer-trajectory differences there.
- The RMT-inspired `c*` implementation is currently heuristic. The code comments, report text, and derivation are not fully aligned; a bootstrap null or a cleaner BBP derivation would be stronger than the current `1.2 * sqrt(gamma)` margin.
- High-batch regimes have large run-to-run variance because they have few optimizer steps at fixed 1B token budget. More seeds or larger token budgets may be needed for 16M-scale claims.
- "Resolved curvature" is shorthand. The diagnostic directly measures reproducibility of singular directions of gradient/momentum matrices, not Hessian eigenvectors. Connecting that to curvature/river-valley geometry still needs either theory or targeted toy experiments.
- A switch can lose even when the metric correctly predicts the LITE regime because training is path-dependent: pure LITE may get useful representation/parameter movement during the early high-LR phase that Muon -> LITE cannot recover after the switch.
- A mid-training metric may answer "is LITE locally reasonable now?" but not "would starting LITE earlier have been better?" These are different causal questions.
- v16 suggests the metric does not yet answer Sadhika's degradation question. At 512K, LITE->Muon switches after c* falls below the crossover threshold, but still loses to pure LITE. So low D1 may mark a lower-alignment regime without implying that the current LITE trajectory has begun to degrade relative to staying LITE.
- Early v17 partial data suggests the checkpoint branch is a useful diagnostic: at the first post-branch eval, switch-to-LITE is slightly ahead of continue-Muon at step 255 (`1.075797` vs `1.079299`). This is not a final claim until the branch reaches 1B tokens.
- v11 suggests "different χ magnitude" should not mean simply lowering χ everywhere. A useful hybrid, if it exists, probably needs per-step/per-layer gating or adaptive rank rather than one global smaller χ.
- v12 makes the lower-global-χ direction low priority. The next hybrid should change *when/where* LITE applies amplification, not just globally reduce χ.

### 4. Hypotheses

- LITE helps when `M_tilde` contains reproducible low-curvature/flat directions; it hurts when its flat subspace is mostly unresolved noise that χ amplifies.
- The best hybrid is probably not a binary Muon/LITE switch, but an adaptive χ or adaptive `d_s` schedule driven by D1(`M_tilde`) or a bootstrap spike count.
- The crossover batch size should shift with model scale; corrected d=12 data should move the curve right if the signal-to-noise/RMT story is right.
- Muon's polar update may actively suppress raw-gradient spike alignment over training, explaining why Adam-warmup D1 and in-training Muon D1 live on different absolute scales.
- An adaptive χ schedule should be most useful at 512K/1M, where fixed LITE wins only slightly and D1(`M_tilde`) is low/declining. However, v10/v11 suggest it cannot be pure time decay or unconditional lower χ; it needs to preserve χ=2 when the diagnostic still says resolved.
- A bootstrap-null threshold may replace both the hand-picked `0.7` and the current heuristic `c*`, especially for momentum-D1 where the EMA changes the null distribution.
- First hybrid test should be schedule-only, not a complicated controller: triangular χ (`1 -> 2 -> 1`) directly tests whether late high-χ hurts after D1 declines. If it helps at 512K/1M, then a D1-driven schedule is worth implementing.
- Early v10 partial result falsifies the simplest "always decay χ late" idea for high-alignment regimes: at 4M and 16M, triangular χ still beats Muon but loses to fixed χ. Any adaptive schedule should probably leave χ high when D1(`M_tilde`) remains moderate/high, rather than decay purely by time.
- Full v10 extends the same pattern to 512K and 1M: pure time decay is not enough even in borderline regimes. If a hybrid wins, it likely needs either a different χ magnitude or a genuinely diagnostic-driven gate.
- v11 falsifies the simplest lower-magnitude variant in the crossover region: fixed χ=1.5 is worse than fixed χ=2 at both 512K and 1M. The next plausible controller should gate `d_s` or χ by D1, not choose a smaller global χ.
- v12 closes the main caveat on χ=1.5: adjacent LR checks do not beat χ=2. Lower global χ is very unlikely to be the right hybrid axis.
- v13 should clarify whether the D1 gate can separate 256K from 512K on the same driver. If 256K D1 overlaps 512K despite different winner/tie behavior, a scalar global gate will be weak and adaptive rank/per-layer logic becomes more plausible.
- v13 does show overlap: the next controller should probably use a band/continuous mapping or adaptive rank, not a binary late-threshold switch.
- A plausible first controller is not "turn χ off late"; it is "warm up as fixed LITE, then use early/mean `c*` to choose a conservative χ or `d_s` band." But because 512K still benefits from χ=2, any controller must avoid suppressing χ in the borderline-win regime.
- v14 tests the minimal binary version of that idea: first-probe `c* >= 0.417` keeps fixed LITE; otherwise χ target becomes 1.0. This is intentionally simple and may be too brittle, but it directly tests whether the calibrated early-D1 separator has operational value.
- v14 is a negative result for training-time binary global χ gating. It classified 256K/512K/1M correctly, but switching 256K to χ=1 after the first probe made it worse than both Muon and fixed LITE. The failure is likely the late trajectory switch, not the diagnostic's classification.
- v15 tests whether adaptive rank is a better axis than adaptive χ: increasing `r_s` should reduce how much unresolved flat subspace gets amplified while preserving χ=2 where useful.
- True Muon -> LITE switching may underperform pure LITE because the advantage is front-loaded: if LITE's main benefit is early exploration/flat-valley movement during large LR, switching after the diagnostic sacrifices the most valuable part.
- True LITE -> Muon switching is the sharper degradation test: if D1 falls late and LITE begins amplifying noise, then switching to Muon after a low-alignment trigger should beat pure LITE. If it does not, the metric predicts final winner but not local optimizer degradation.
- Full v16 favors the path-dependence hypothesis: Muon -> LITE switches at the intended first probe in the 512K/1M LITE regimes but remains close to Muon and worse than pure LITE. The diagnostic tells which pure trajectory to prefer better than it tells how to splice trajectories.
- StreamingMuon/SPECTRA may be a better vehicle than hard Muon/LITE switching for testing adaptive spectral control: a LITE-like transform would be `f_i=1` on sharp/top directions and `f_i=chi` on flat/tail directions. Because the cached basis columns are warm-started and not guaranteed semantically ordered, any top-rank mask should be defined by current `sigma` top-k/threshold rather than column index.
- When restarting sigma-search, the fair ranking baseline should be identity StreamingMuon under the same `fallback_orthogonality_tol` as every candidate. Loose tolerance may preserve candidate ordering for short sweeps, but it is not safe for interpreting absolute wins over native Muon.
- The 2026-04-29 DDP8 d8/32-step probe suggests `tol=0.01` and `tol=0.02` are indistinguishable at short horizon (`1.857972` vs `1.857955` BPB), so `0.02` may be acceptable for short candidate screening. Long-horizon claims should still default to `0.01` or explicitly compare identity at the same tolerance.
- Very short or compressed training schedules can exaggerate StreamingMuon-vs-native differences. A 256-step compressed schedule produced a larger streaming advantage (`+0.0187` BPB native-streaming), while the 1024-step schedule shrank the gap to `+0.00145`. Use at least the 1024-step DDP8 probe before calling identity/non-identity candidate differences real.
- The d12/1B match strongly supports using StreamingMuon identity `tol=0.01` as the operational Muon-like baseline for sigma-search. Candidate improvements should still be reported against same-tolerance identity, because identity is the direct control for the streaming approximation and candidate-state machinery.
- Top-1 damping is the simplest test of the sharp-direction LR-bound story: set the current largest `sigma` direction to `alpha=0.5` and leave all other directions Muon-like. If the story is right, same-LR performance may be similar or slightly worse, but the best-LR frontier should move upward because non-top directions can use a larger global LR.
- A fixed LR grid is the wrong way to test top-1 damping. The sweep should be adaptive: start with a small grid around known StreamingMuon LRs, then expand only when the best point is on a boundary or near-boundary. If high LR is already clearly worse, do not continue upward.
- The first official top1-damp DDP8 sweep supports the LR-bound mechanism at medium/large batch but not at 128K. At 128K, top1 is a wash/slightly worse than identity (`-0.000072` BPB). At 1M, top1 wins by `+0.001818` BPB with best LR shifting from identity `0.01` to top1 `0.02`. At 8M, top1 wins by `+0.015381` BPB at the same best LR `0.02`. This is currently a one-seed sigma-transform observation, not a final multi-seed claim.
- Do not compare raw BPB from DDP `run_eval.py` top1/identity runs directly against old single-GPU v9/v18 Muon/LITE baselines. The validation stream differs before training: DDP step0 is `3.155445`, single-GPU step0 is `3.142474`. Top1 claims should be phrased relative to same-driver StreamingMuon identity unless a same-driver native DDP control is explicitly run.
- Exact DDP controls now show StreamingMuon identity is close to native DDP Muon at 128K (`+0.000276` BPB) and 1M (`+0.001604` BPB), but not at 8M (`-0.020191` BPB, streaming better). Therefore top1 at 1M is a clean same-driver improvement over a validated identity baseline; top1 at 8M remains confounded until the 8M identity gap is explained.
- 1M identity diagnostics suggest SCQR/tol is not causing a large mismatch: pure QR, 2-iter SCQR, and 2-iter pure QR all stay around native DDP performance. The unresolved numerical/algorithmic question is specific to 8M.
- For fixed-token-budget batch-size sweeps, fixed warmup step count is the wrong schedule anchor. It gives different warmup token fractions at different batch sizes: too short at 128K and too long at 8M. Use fixed schedule fraction instead.
- Official sigma-search schedule recipe: `warmup_steps=round(0.05 * max_steps)`, `warmdown_ratio=0.65`, `final_lr_frac=0.05`. This matches the clean v9/v18 schedule family and makes different batch sizes comparable under a fixed total token budget.
- Boundary LR policy: if the best LR is on a grid edge, do not interpret the winner as tuned. Extend the grid outward on that side until the best point is interior or the next point is clearly worse. This applies both to LITE/Muon baselines and to StreamingMuon sigma-transform candidates.
- For StreamingMuon top1, the next rigorous step is conditional: exact native DDP controls must first show identity is close under the same top1 recipe. If identity is not close, top1 alpha sweeps are blocked and the priority becomes numerical diagnostics (`pure_qr`, two streaming iterations, and input normalization). If identity is close, alpha becomes a real hyperparameter and should be swept around `0.5` with LR re-tuning.
- The clean StreamingMuon infra now logs Muon-relevant tensors from the actual optimizer step when requested: post-allreduce gradient, updated momentum buffer, Nesterov-corrected momentum input, and streaming sigma. This avoids the old ambiguity where diagnostics could accidentally inspect a different object from the optimizer update.
- Hessian diagnostics should remain opt-in and best-effort. In practice, higher-order autograd through attention kernels can fail, and even successful HVP probes are too expensive for routine 1B-token sweeps.

## What the two methods are really measuring

Both D1 (split-batch SVD alignment) and D2 (BBP/MP spike detection) are asking the same question with different priors:

> Is the top-k singular subspace of $G$ *reproducible* across independent samples from the data distribution, or is it an artifact of the specific minibatch?

- **D1** measures reproducibility *directly*: sample two $G_A, G_B$; check whether their top directions match.
- **D2** measures reproducibility *indirectly*: if the noise distribution of $G$ looks like an MP bulk + a few spikes, the spikes must be population-level (BBP theorem tells us below threshold they'd be unresolvable).

D1 is data-efficient but needs two grad passes. D2 needs only one grad but requires a calibrated noise model.

## Why LITE amplifies when the spikes *aren't* resolved

Muon sends all σ → 1. LITE sends σ → 1 on the top-d_s (sharp), and σ → χ>1 on the rest (flat). If the flat subspace contains un-resolved population directions (bulk that includes real curvature), LITE amplifies the wrong modes. That's the small-batch regression.

**Key insight I want to test:** the d_s cutoff in LITE plays the role of a rank threshold. If we set d_s adaptively to the BBP-spike count, LITE should stop mis-classifying "almost-spike" directions as flat. This is a natural bridge from D2 back into the LITE algorithm itself — not just a switcher, but a repair.

## Unifying D1 and D2

D1 alignment $c_i = |u_A^{(i)\top} u_B^{(i)}|$ under random matrix theory:
- For a true spike above BBP threshold: $\mathbb{E}[c_i^2] = \frac{1 - \gamma/s^2}{1 + \gamma/s}$ where $s$ = signal SNR, $\gamma = n/N$. Approaches 1 as SNR grows.
- For a bulk direction: $\mathbb{E}[c_i^2] \approx 1/r$ where $r$ is the rank — vanishes for wide matrices.

So Sadhika's 0.7 threshold corresponds roughly to "SNR > ~1.5 above BBP threshold". **We can replace it with a data-dependent one**: the threshold $c^* = \sqrt{(1 - \gamma/s^*)/(1 + \gamma/s^*)}$ evaluated at the critical SNR $s^* = \sqrt{\gamma}$ (BBP point). Below this is bulk, above is resolved spike. This is cleaner and falls out of RMT.

## What to validate empirically
1. Does $c_i$ show a visible bimodal distribution in real training? (spikes near 1, bulk near 0)
2. Does the fraction of $c_i > c^*$ (data-dependent) predict LITE-vs-Muon winner as well as fraction $> 0.7$?
3. Is D1 consistent with D2 at each (bsz, step)? I.e., does spike count from D2 ≈ number of $c_i > c^*$ in D1?
4. When does the picture shift during training? Early training: grads are crisp (large scale → high SNR). Late training: grads shrink, SNR drops, maybe LITE should turn off late.

## Practical subtleties

- **Per-layer vs aggregated.** Sadhika's framing is batch-level, but Muon/LITE operates per parameter matrix. The diagnostic should be per-layer; aggregate via mean or median over layers. Watch for one layer dominating.
- **Which parameters?** Only matrix params that LITE touches (transformer attn/MLP weights). Exclude embeddings/scalars.
- **Split size.** Half-batch is cleanest but doubles memory. Alternative: microbatch-split within the same accumulation — pool two microbatches into $G_A$, two into $G_B$. Costs basically nothing since we're already accumulating.
- **Momentum object.** Raw per-step gradient is still useful for comparability with Sadhika's original proposal, but LITE/Muon project `M_tilde`. The current best practice is to report both: raw-gradient D1 for continuity and split-half scheduled `M_tilde` D1 for algorithm relevance.
- **Rank of $d_s$.** In the team's LITE config so far, d_s=32. If the spike count from D2 is much less than 32, LITE is projecting out "fake" spikes. If much more, LITE is flattening real curvature. Should report the gap.

## Why "large batch" helps in nanochat's numbers
Large batch (8M tok) = lower gradient noise variance by factor ~64× vs small (128K). MP-bulk radius scales as $\sqrt{n/N}$. At 8M tok with matrix param of size e.g. 768×768, $n/N$ is tiny, bulk is near zero, everything useful is resolved — LITE amplifying "flat" directions is essentially amplifying resolved near-zero-curvature directions, which is what it's designed to do. At 128K, bulk is thick, bulk directions are mislabeled as flat, LITE pushes on noise.

## Risks to the story
- If the crossover bsz depends strongly on model scale / d, the diagnostic needs to be tested at d>8 before we claim it's universal.
- If d=8 is too small for MP bulk structure to be visible (768-dim params), BBP may be hard to apply cleanly — there's essentially no bulk. D1 might still work since it's finite-sample.
- LITE's χ=2 is a tuning parameter. Bigger χ amplifies more; there may be a χ threshold where LITE wins on ALL batch sizes but loses tangentially elsewhere. Should pin χ=2 (Zhu et al. default) for the diagnostic story.

## Open questions I'd like to answer with this campaign
1. Is there a single scalar (maybe just $r_{\text{eff}}$) that predicts winner across all (bsz, step)?
2. Does the winning rule change through training, suggesting an adaptive switch?
3. Can LITE with BBP-adaptive $d_s$ strictly dominate both fixed-$d_s$ LITE and Muon?
