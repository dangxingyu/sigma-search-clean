# Experiment Plan — LITE vs Muon Switching Diagnostics

## Current clean-handoff plan as of 2026-05-02

Status checkpoint:
- v42 completed the core `c=1` vs `c=0.5` d8 / 0.4B-token sweep at `{262K,1M,4M}`. Treat the current evidence as: `262K` identity slight win, `1M` effectively tie/tiny `c=0.5` win only after LR extension, `4M` strong `c=0.5` win.
- v43 completed the first dense dynamics run at `4M` with metrics every step and Hessian top-4 probes every 24 steps. Use `results/metrics_v43_4m_best/` for the current dynamics sanity plots.
- v44/v45 completed the transition sweep: `2M` favors identity after explicit `lr=0.16` boundary closure, while `8M` strongly favors `c=0.5`. Use `results/sweep_catalog/v44_v45_transition_summary.*`.
- v46 completed the 2M dynamics sweep: identity and Top-Aware `c=0.5`, both at `lr=0.08`, metrics every step, Hessian top-4 every 48 steps. Use `results/metrics_v46_2m_best/`.
- v47 closed the 1M `c=0.5` high-LR edge: `lr=0.16` is worse than `0.08`, so the current 1M best row is closed.
- v48 completed 8M dynamics: identity and Top-Aware `c=0.5`, both `lr=0.02`, metrics every step, Hessian top-4 every 12 steps. Use `results/metrics_v48_8m_best/`.
- v49/v50 completed the 2M seed check at `lr=0.08`: seeds `{42,43,44,45,46}` are a tie/noisy transition, mean `c=0.5 - identity = +0.000507 ± 0.004183` SEM.
- v51 confirmed the 4M Top-Aware win across seeds `{42,43,44}`: mean `c=0.5 - identity = -0.025240 ± 0.004110` SEM.
- v52-v55 completed the 1M seed check with high-LR closure: seeds `{42,43,44}` are near-tie, mean `c=0.5 - identity = +0.001850 ± 0.002798` SEM.
- v56/v57 completed the 262K seed check: seeds `{42,43,44}` are near-tie, mean `c=0.5 - identity = -0.000613 ± 0.001083` SEM.
- Current next useful runs are optional robustness checks, not blockers: either add more seeds in the 262K/1M/2M transition band, or run dynamics metrics at 1M/262K if we need mechanism rather than winner classification.

Immediate standalone-repo priorities:
- Use the clean method set: `top_aware_muon` and `streaming_identity`. Keep native Muon/LITE code as deprecated targeted validation controls, not as default sweep methods.
- For handoff optimizer-quality sweeps, use `scripts/run_handoff_sweep.sh`. Keep metrics off by default, start from an LR grid such as `{0.005,0.01,0.02,0.04}`, and leave `ADAPTIVE_LR=1` so boundary LR optima are extended automatically before interpreting winners.
- For the new dynamics/logging study, run `streaming_identity` (`c=1`) versus Top-Aware Muon `top_k=1, alpha=0.5` (`c=0.5`) at batches `{262144,1048576,4194304}`.
- Treat `alpha` as the document's T-Muon coefficient `c`. Do not spend default compute on `0.75/0.25` or a duplicate `top_aware alpha=1` until the two-point test gives a reason.
- If the `1M` batch does not show an obvious Top-Aware improvement over identity, pivot the medium/large batch probe to `{2097152,8388608}` rather than expanding alpha.
- Treat `262144` as the d8 critical batch. Do not insert an extra 512K point into this specific no-tuning metrics grid.
- Use the d8 Chinchilla-style token budget `402,653,184` tokens, about `0.4B`, with nanochat LR scaling and no additional LR/alpha tuning.
- Use `scripts/run_d8_metrics_grid.sh` for the canonical grid. It records cheap StreamingMuon metrics every step and Hessian probes every 50 logged steps with math SDPA forced.
- Use corrected logging semantics: raw momentum-buffer metrics are not logged, `train/loss` is raw optimizer-step mean CE, and Hessian probes are global selected-matrix-subspace Lanczos HVPs over all normal transformer matrix weights on a representative rank0 microbatch from the same optimizer step.
- Canonical StreamingMuon metrics should use no explicit SVD: reuse cached `sigma` and basis by default. Do not enable exact per-module SVD, component saving, or legacy split-SVD alignment unless explicitly auditing those diagnostics.
- Use `results/sweep_catalog/` as the organized source for completed sweep rows; rebuild with `python scripts/build_sweep_catalog.py` whenever curated CSV/JSON summaries change.

Derived from `guidance.md` (Sadhika's two-diagnostic framework).

## Current plan as of 2026-04-29, switching-priority pivot plus StreamingMuon DDP restart

### Objective
Build a rigorous d=8 / 1B-token study of when vanilla Muon, fixed-χ LITE, and optimizer switching should be used. The diagnostic under test is Sadhika's split-batch SVD alignment, but the current working hypothesis is that the SVD should be taken on the optimizer input direction `M_tilde` (Nesterov-corrected momentum), not the instantaneous raw gradient. The current priority is no longer "find the best LITE variant"; it is to answer whether a metric-driven single switch between Muon and LITE helps, and why preliminary Muon -> LITE switching appears worse than pure LITE.

### Constraints
- Do not cancel or kill SLURM job `29702470`; it holds the active 8×B200 allocation.
- Do not use Modal for new experiments.
- Avoid weak smoke-only claims. Real claims require normal-scale training: d=8, ~1B tokens, matched LR, and preferably ≥3 seeds.
- Treat `sweep_results_v8/` as bug-affected d=12 exploratory data only: original DDP drivers used effective batch = `world_size × nominal`.

### Completed current experiment
- `sweep_results_v9/` is complete: d=8, 1B tokens, batch sizes `{512K, 1M, 4M, 16M}`, 2 optimizers, 3 seeds = 24 full single-GPU runs.
- v9 used matched best-known LRs from prior sweeps: Muon/LITE `{512K: 0.01/0.01, 1M: 0.01/0.02, 4M: 0.04/0.04, 16M: 0.04/0.04}`.
- v9 primary telemetry records split-half `M_tilde` alignment, where `M_tilde = (1-mom)g + mom*m_new`, on the LITE projection side: tall matrices compare right singular vectors (`V`), wide matrices compare left singular vectors (`U`).
- Raw momentum-buffer alignment is saved as `d1_probes_momentum_buffer`; Sadhika-style left-U `M_tilde` alignment is saved as `d1_probes_momentum_left`.
- `sweep_results_v18_128k_clean/` is complete and LR-closed on the low side: d=8, 1B tokens, 128K batch, optimizers `{Muon,LITE}`, LRs `{0.005,0.01,0.02,0.04}`, seeds `{42,43,44}`.
- 8-GPU StreamingMuon DDP smoke/probe is complete: `search_evals/ddp8_streaming_smoke_20260429_111021/` and `search_evals/ddp8_streaming_d8_probe_20260429_111919/`.
- Native Muon vs StreamingMuon identity `tol=0.01` DDP8 match test is complete: `search_evals/ddp8_streaming_native_match_20260429_112759/`.

### v9 outcome
- LITE wins at all tested v9 batch sizes, with positive Δ = Muon BPB - LITE BPB: `512K +0.0024`, `1M +0.0048`, `4M +0.0124`, `16M +0.0253`.
- Absolute Muon-trajectory D1(`M_tilde`, c*) increases with Δ: last-probe values `512K 0.354`, `1M 0.375`, `4M 0.792`, `16M 1.000`.
- Optimizer-trajectory contrast is not useful: `D1(LITE) - D1(Muon)` is near zero or wrong-signed despite LITE wins.
- The hard `c > 0.7` threshold carries useful dynamic range for `M_tilde`; the current `c*` threshold is often too permissive at high batch.
- Side-aware alignment is required at low batch: at 512K Muon last probe, LITE-side c* is `0.354` while left-U c* is `0.062`.

### Immediate next steps
- Mainline method set is narrowed. For new sweeps, default to `streaming_identity`, `native_muon`, `native_lite`, and `top_aware_muon` only. Do not include `streaming_lite` unless explicitly requested; it is a same-driver sanity check, not a required baseline and not a replacement for native LITE.
- Clean infra now has opt-in no-SVD metric logging in `run_eval.py`. For future Top-Aware/StreamingMuon sweeps that need diagnostics, add for example `--metrics-every 1 --metrics-top-k 4 --metrics-hessian-every 50`. Use Hessian flags only for targeted dynamics probes because HVP is expensive and kernel-dependent.
0. Active smaller-batch fair-comparison run: v26 is running under `search_evals/v26_small_bsz_streaming_fair_20260430`. It compares same-driver StreamingMuon `identity`, `lite_chi2_rs01`, and `top1pm_a05` at global batches `64K` and `32K`, seed `42`, LR grid `{0.005,0.01,0.02}` with boundary extension. All StreamingMuon variants use pure Householder QR and two streaming iterations per step. This is the current source for the small-batch Muon/LITE/top1 question.
   Escalation criterion for d12/3B: only start a larger d12 / ~3B-token run if the small/medium evidence is internally consistent after LR closure: at large batch, `top1pm_a05` beats `lite_chi2_rs01`, which beats `identity`; at small batch, `identity >= lite_chi2_rs01` within tuned LR. If the best LR is on a boundary or the ordering is mixed, first extend LR or add seeds instead of scaling.
   Prepared but not launched: `run_v27_d12_3b_streaming_fair_on_node.sh`. Defaults are depth `12`, tokens `3221225472`, batches `{1M,8M}`, seed `42`, LR grid `{0.005,0.01,0.02,0.04}`, same three StreamingMuon methods, pure QR, and two iterations. Launch only after the gate is satisfied.
   Active supervisor: `run_v26b_large_then_maybe_v27_on_node.sh` is waiting for v26 to finish. It then runs v26b `{1M,8M}` same-driver comparison and uses `decide_scaleup_gate.py` to launch v27 only if the gate passes.
1. d12 StreamingMuon identity baseline check is complete: at d12 / seq1024 / 8GPU / 4096 steps / global batch 262K, native Muon final BPB is `0.866625` and StreamingMuon identity `tol=0.01` is `0.866802`, gap `-0.000177` native-streaming. Treat strict identity StreamingMuon as a usable Muon-like baseline for sigma-search, while still reporting same-tolerance identity beside every candidate.
2. Fold v18/v18b into the switching summary as the clean 128K point. The low-LR boundary is closed: both Muon and LITE choose `lr=0.01`, while `lr=0.005` is worse. The tuned result is an effective tie, not a robust Muon win.
3. For StreamingMuon sigma-search, use 8-GPU DDP through `run_eval.py` with strict fallback (`--fallback-ortho-tol 0.01` or `0.02`). The d8/32-step smoke validates launch, grad accumulation, and candidate state, not long-horizon optimizer quality.
4. Before expensive sigma-search candidate claims, include identity StreamingMuon under the same tolerance as every candidate. The native-vs-streaming identity test says 1024-step d8 `tol=0.01` matches native Muon within `0.00145` BPB; the active d12/1B run will check whether that survives a larger model/horizon.
5. Active sigma-transform experiment: test the top-1 sharp-direction damping hypothesis. Candidate `candidates/top1_damp_alpha05.py` sets `f(sigma_argmax)=0.5` and leaves all other directions at `1.0`. Official recipe is d8 / ~1.07B tokens / batch sizes `{128K, 1M, 8M}` with schedule `warmup_steps=round(0.05 * max_steps)`, `warmdown_ratio=0.65`, `final_lr_frac=0.05`, strict `fallback_ortho_tol=0.01`, and adaptive matrix-LR expansion from `{0.01, 0.02, 0.04}`. Compare top1 first against same-driver StreamingMuon identity. Do not compare raw DDP `run_eval.py` BPB against single-GPU v9/v18 native BPB until exact DDP native controls validate the identity baseline.
   Handoff update: the top-1 idea is now named **Top-Aware Muon** in `candidates/top_aware_muon.py`. Current clean recipe fixes `top_k=1` and sweeps `alpha`; `top_k` remains in code only for explicit ablations. The reusable sweep runner is `run_top_aware_muon_sweep.py`; it sweeps `batch x alpha x lr` and includes baselines `streaming_identity`, `streaming_lite`, `native_muon`, and `native_lite`.
   New immediate Top-Aware alpha experiment: run v28/v29 after active v26 finishes. v28 fills the missing `128K, alpha=0.75` LR sweep. v29 tests `256K` with alpha `{0.5,0.75,0.875}` plus baselines. Working hypothesis: optimal alpha is monotone non-increasing with batch size; identity is alpha=`1`, so 128K should prefer alpha near `1`, while 256K should begin moving lower.
6. Autonomous 8h controller: `run_autonomous_8h_controller.py` waits for v18b, closes the 128K LR boundary, runs exact native DDP controls at top1 identity best-LR points, then either runs identity diagnostics (`pure_qr`, `pure_qr_2iter`, `scqr_2iter`, `input_normalize`) if StreamingMuon identity is not close, or runs the v24 top1 alpha sweep at `{1M,8M}` for `alpha={0.25,0.5,0.75}` reusing existing alpha=0.5 data.

### Next experiments
- **v10 hybrid χ schedule ablation completed:** triangular χ (`1 -> 2 -> 1`) was tested at d=8, 1B tokens, bsz `{512K, 1M, 4M, 16M}`, 3 seeds. It beats Muon at all bsz but loses to fixed χ=2 at all bsz: hybrid-fixed deltas are `512K -0.0003`, `1M -0.0006`, `4M -0.0038`, `16M -0.0109`. Conclusion: time-only late χ decay is not the right hybrid.
- **v11 fixed lower-χ crossover ablation completed:** χ=1.5 was tested at `{512K, 1M}`, d=8, 1B tokens, 3 seeds, same LR as v9 fixed LITE. χ=1.5 still beats Muon but loses to fixed χ=2 at both bsz: χ=1.5 minus χ=2 BPB is `+0.0008` at 512K and `+0.0011` at 1M. Conclusion: the simple "χ=2 over-amplifies at crossover" hypothesis is not supported under current LRs.
- **v12 χ=1.5 LR-check completed:** χ=1.5 was tested at 512K with LRs `{0.005, 0.01, 0.02}` and at 1M with LRs `{0.01, 0.02, 0.04}`, 3 seeds each, d=8, 1B tokens. Best χ=1.5 rows remain worse than χ=2: 512K best `lr=0.02` is `+0.0004` BPB worse than χ=2; 1M best `lr=0.02` is `+0.0011` BPB worse.
- **v13 clean 256K baseline completed:** local v9-style Muon vs fixed χ=2 LITE at 256K, lr `0.01`, 3 seeds, d=8, 1B tokens, with `M_tilde` D1 telemetry. Result: Muon `0.8922`, LITE `0.8923`, Δ `-0.0001 ± 0.0005`; effectively tie / slight Muon.
- **Gate calibration implication:** 256K and 512K have overlapping late hard D1(`M_tilde`, `c>0.7`) at `0.125`, so a single late hard threshold is not enough. First/mean `c*` separates them better (`256K mean 0.346`, `512K mean 0.400`, `1M mean 0.537`) but the crossover band is narrow.
- **Gate calibration result:** `analyze_gate_calibration.py` shows the largest 256K/512K separation comes from first-probe `c*` (threshold ≈ `0.417`, margin `+0.125`). Late hard `c>0.7` has zero margin and should not be used as the gate.
- **v14 first-D1-gate completed:** first-probe `c*` gate at `{256K, 512K, 1M}`, 3 seeds, d=8, 1B tokens. The gate classified correctly (`256K -> χ=1`, `512K/1M -> χ=2`) but did not improve training: 256K gate BPB `0.8935` is worse than Muon `0.8922` and fixed LITE `0.8923`; 512K/1M match fixed LITE.
- **v15 sharp-rank sweep completed:** fixed χ=2 with larger sharp rank `r_s={0.2,0.5}` at `{256K,512K}`, 3 seeds, d=8, 1B tokens. Result: global larger `r_s` is not a robust fix; `r_s=0.2` slightly helps 256K but slightly hurts 512K, while `r_s=0.5` hurts 512K more.
- **v16 true one-switch optimizer test completed:** Muon -> LITE and LITE -> Muon were run at `{256K,512K,1M}`, 3 seeds, d=8, 1B tokens, threshold `0.417`. Main result: switching does not beat the better pure optimizer. Figure: `v16_optimizer_switch_analysis.png`.
- **v17 checkpoint-branch counterfactual completed:** one targeted `1M, seed=42` branch from the same Muon step170 checkpoint. Switching the branch to LITE beats continuing Muon by `+0.001185` BPB, but remains worse than pure LITE's mean at 1M. Figure: `v17_checkpoint_branch_analysis.png`.
- **v18/v18b 128K clean sweep completed and low-LR closed:** bsz `128K`, opt `{Muon,LITE}`, LR `{0.005,0.01,0.02,0.04}`, seeds `{42,43,44}`, d=8, 1B tokens. Best LR is `0.01` for both; Muon `0.89370`, LITE `0.89358`, delta Muon-LITE `+0.00012`. `lr=0.005` is worse for both, so no `0.0025` extension is needed. Figure: `v18_128k_clean_analysis.png`; consolidated LR-sweep figure: `optimizer_lr_sweeps.png`.
- **StreamingMuon DDP8 smoke/probe completed:** d4 smoke covered identity `tol={0.01,0.02,0.05}`, grad-accum=2, pure QR, and tikhonov adaptive. d8/seq1024 32-step probe covered identity `tol={0.01,0.02}` and tikhonov adaptive at global batch 262K. All `run_eval.py` cases completed with no errors; base_train wrapper also completed one 8GPU step plus checkpoint save after wrapper fixes.
- **Native-vs-Streaming identity match completed:** d8/seq1024/global batch 262K/8GPU, seed 42. At 1024 steps, native Muon final BPB `0.995475`, StreamingMuon identity `tol=0.01` final BPB `0.994029`, delta native-streaming `+0.001447`. Conclusion: `tol=0.01` identity is close enough to native Muon for sigma-search baselines, but compressed 256-step schedules can exaggerate differences.
- **Corrected d=12 StreamingMuon scaling completed:** launcher `run_ddp8_streaming_vs_native_d12_1b_on_node.sh`, seed `42`, depth `12`, seq `1024`, global batch `262144`, `4096` steps, eval every `512`, matrix LR `0.02`, identity `fallback_ortho_tol=0.01`. Final native Muon `0.866625`; StreamingMuon identity `0.866802`; final native-streaming delta `-0.000177`. Prior v8 is a bug artifact and should only motivate, not support, model-scale claims.
- **Threshold calibration:** compare hard `0.7`, current `c*`, and a bootstrap-null threshold on the same v9 buffers; current evidence says `c*` is over-permissive for momentum-D1.

### v16 mainline: true one-switch optimizer experiment

**Research question.** Does a split-batch `M_tilde` statistic make a one-time Muon/LITE optimizer switch useful, or does switching lose because the optimizer path matters?

**Implementation requirement.** This must not be another `χ=1` proxy. The switch runner should use one optimizer object with shared AdamW state and shared matrix momentum buffers, but matrix groups must execute:
- native Muon phase: Polar update only, with nanochat Muon aspect-ratio LR scaling;
- LITE phase: current LITE projection/amplification update with `χ=2`, `r_s=0.1`.

**Core sweep.**
- Batch sizes: `256K`, `512K`, `1M`.
- Seeds: `42, 43, 44`.
- Directions: `muon_to_lite`, `lite_to_muon`.
- Scale: d=8, 1B tokens, local single-GPU jobs, same schedule as v9/v13/v14.
- LR: use the fixed-LITE LR for the switch run (`256K=0.01`, `512K=0.01`, `1M=0.02`) so the comparison to pure LITE is direct; note that `1M` Muon phase is not Muon's best LR and may need a follow-up LR check if promising.

**Switch rule.**
- Metric: side-aware split-half `M_tilde` alignment, top-16, `frac_above_c_star_median`.
- Threshold: start with the calibrated crossover threshold `0.417`.
- `muon_to_lite`: switch once when metric rises above threshold; otherwise stay Muon.
- `lite_to_muon`: switch once when metric falls below threshold; otherwise stay LITE.

**Success criteria.**
- For `muon_to_lite`, success means beating pure LITE, not merely beating Muon.
- For `lite_to_muon`, success means beating pure LITE after the metric says LITE has entered a low-alignment/degradation regime.
- If neither direction beats the better pure optimizer, the conclusion is that the metric predicts the winner/regime but is not sufficient for mid-training optimizer switching.

## Central question
When does Muon-LITE (χ>1 amplification on flat directions) beat vanilla Muon, and when does it lose? Produce a diagnostic that predicts the switching point from a short warm-up run, not from a full training sweep.

## What we already know (pre-ralph-loop)
Three 1B-token d8 runs at lr=0.02 (from memory `finding_spectra_wins_at_1B.md`):

| setting | tokens/step | Muon BPB | LITE BPB | Δ (Muon−LITE) | winner |
|---|---|---|---|---|---|
| small | 128K | 0.871 | 0.915 | -0.044 | Muon |
| medium k=1 | 512K | 0.874 | 0.905 | -0.031 | Muon |
| large k=16 | 8.4M | 0.889 | 0.875 | +0.014 | **LITE** |

Pattern consistent with Sadhika's theory: large batch → curvature spikes resolved → LITE wins.
Two deficiencies in these runs:
- LR not matched per optimizer (we know native peaks at 0.02; LITE peak unknown per bsz).
- No split-batch or spectrum diagnostics were recorded — the theory is un-validated, only the outcome is.

## Work units

### WU-1: Diagnostic implementation
- **D1 — Split-batch SVD alignment** (`diagnose_split_batch.py`). For a given (model checkpoint, optimizer, batch): split batch → two grads per layer → SVD each → compute per-layer $c_i = |u_A^{(i)\top} u_B^{(i)}|$ for top-k singular directions. Report: per-layer array + aggregated counts (>0.7, <0.3, middle).
- **D2 — Spectrum + effective rank** (`diagnose_spectrum.py`). Single gradient per layer → full SVD → plot spectrum, fit bulk (Marchenko-Pastur reference), identify spikes, compute $r_{\text{eff}} = (\sum\sigma_i)^2 / \sum\sigma_i^2$. Report per-layer $r_{\text{eff}}$ + spike count.
- Both diagnostics must run in <60s per snapshot on 8×B200 so they can be called during training.

### WU-2: Batch-size scan (existence of crossover)
Batch sizes: 128K, 256K, 512K, 1M, 2M, 4M, 8M tokens/step (7 points, log-spaced). For each:
- Native Muon: lr ∈ {0.005, 0.01, 0.02, 0.04}. Pick best.
- LITE (χ=2, d_s=32): lr ∈ {0.005, 0.01, 0.02, 0.04}. Pick best.
- 1B tokens total. d=8.
- Record: final BPB, loss curve, wall time, + diagnostic snapshots at steps {100, 500, 2000, 5000}.

Total runs: 7 bsz × 2 opt × 4 lr = 56. At ~20 min each on 8×B200 (node has ~2 days): 56 × 20 min = ~19 h compute if sequential — tight but feasible. We'll prioritize and drop LRs that are clearly off-peak after the first two bsz.

### WU-3: Diagnostic validation
For every (bsz, opt, lr) run in WU-2:
- Snapshot gradient stats at step 500 (after warm-up).
- Compute: (a) fraction of top-32 directions with c_i > 0.7 (D1), (b) per-layer $r_{\text{eff}}$ (D2).
- Correlate diagnostic → Δ(Muon − LITE) final BPB.
- Goal: find a threshold on D1/D2 that cleanly predicts the winner from step 500.

### WU-4: Threshold principledness
Sadhika flagged 0.7 as arbitrary. Test alternatives:
- Derive expected $c_i$ under pure-noise (Marchenko-Pastur + random unitary). Use that as null threshold.
- Ideally D2 (BBP) gives a data-dependent threshold that agrees with D1 in practice.

### WU-5: Holistic report
Cover: theory recap, diagnostic methods, crossover scan, validation correlation, principled threshold, open questions. Figures: bsz × BPB curve, c_i histogram vs bsz, $r_{\text{eff}}$ vs bsz, diagnostic-vs-outcome scatter, spectrum plots for 3 bsz regimes.

## Execution order
1. Write `new-thoughts.md` (my reading of guidance.md). [iter 1] ✅
2. Implement D1 (`diagnose_split_batch.py`). [iter 1–2] ✅
3. Implement D2 (`diagnose_spectrum.py`). [iter 2] ✅
4. Sanity-check both at one (bsz, opt) on existing checkpoint or short run. [iter 2–3] ✅ (probe_grads.py, warmup=100 and =500)
5. Launch WU-2 runs in staged batches on the 8×B200 node using `srun --overlap --jobid=28829013`. [iter 3+]
6. Collect results into `experiment-log.md` with figures inline. [rolling]
7. Analyze correlation, write principled-threshold section. [late iterations]
8. Write holistic report. [final iteration]

## WU-2 LR-sweep execution recipe (for iter 3)

**Goal:** verify memory's Muon-vs-LITE deltas at matched LR per optimizer, plus fill in the crossover region (1M–4M tokens/step).

**Shapes:** d=8, seq=1024, model_dim=512. Use `run_native_muon.py` for Muon, `run_lite.py` for LITE (default χ=2, d_s=4).

**Grid (minimum viable):**
- bsz ∈ {128K, 1M, 8M} — 3 points (two known + one crossover)
- opt ∈ {Muon, LITE} — 2 points
- lr ∈ {0.01, 0.02, 0.04} — 3 points around known peak
- 1B tokens each (steps = 1e9 / bsz)
- seed = 42

**Total: 18 runs.**

**Device-batch / grad-accum per bsz (single-GPU, dev_bsz=16, seq=1024):**
| bsz (tok) | micros per step | num_iters | est. wall (1 B200) |
|---|---|---|---|
| 128K | 8 | 7630 | ~25 min |
| 1M | 64 | 1000 | ~25 min |
| 8M | 512 | 125 | ~30 min |

For 8M bsz we can optionally scale to multi-GPU with DDP (world=4, dev_bsz=16, grad_accum=32) to reduce wall time to ~10 min per run, if DDP works with run_native/run_lite.

**Parallelism:**
- Node has 8 B200s. Small-bsz runs take 1 GPU each → 8 in parallel.
- Phase 1: 128K × 2 opt × 3 lr = 6 runs. Launch all in parallel on 6 GPUs. ~25 min wall.
- Phase 2: 1M × 2 opt × 3 lr = 6 runs. Parallel on 6 GPUs. ~25 min wall.
- Phase 3: 8M × 2 opt × 3 lr = 6 runs. Either sequential on 8 GPUs (DDP) or parallel on 6 separate GPUs (each with grad_accum=512, slower). ~30-60 min wall.

**Total wall estimate: ~90 min.**

**Output per run:** JSON with `{val_bpb_final, val_bpb_best, train_loss, args}` — already produced by existing drivers.

**Orchestration script to write in iter 3:** `run_lrsweep.sh` — sbatch/srun launcher that spawns all 18 runs with unique output filenames.

**Decision rule (post-run):**
- For each (bsz, opt) pair, pick best-LR val_bpb.
- Compute Δ = Muon_best − LITE_best at each bsz.
- Correlate Δ with D1 `frac>0.7` diagnostic from `probe_warmup*.json`.
- Winning condition for diagnostic: sign of Δ agrees with diagnostic's LITE/Muon call at every bsz.

## Resources
- Node: SLURM job `28829013` on `c1006a-s25`, 8×B200, ends 2026-04-25T10:42. ~2 days remaining — most of WU-2 must fit in this window.
- If job ends before WU-2 done: queued job `29702470` should pick up; re-verify its nodelist when it starts.
- Venv: `nanochat/.venv/bin/activate`.

## Out of scope
- Changing the LITE algorithm itself.
- Distributed variants of LITE (already have single-rank version).
- Other optimizers (SPECTRA, StreamingMuon) — scope is LITE vs native Muon only.
