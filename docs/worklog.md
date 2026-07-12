# Experiment Worklog

A running lab notebook of **what we tried, why, what happened, and what we learned**
for each hypothesis — the experimental reasoning, not the code (the code is
documented elsewhere). Newest entries at the bottom of each section.

**How to add an entry:** date it, state the *step* (what was tried), the *result*
(numbers, honestly), and the *takeaway* (what it means + next step).

---

## H1 — Self-supervised pretraining for semi-supervised segmentation

**Question:** does JEPA pretraining beat a from-scratch supervised model when labels
are scarce? Compare across labeled fractions {0.1, 0.5, 0.75, 1.0}.

### 2026-07-12 · Step 1 — First real run (Kaggle). Negative result.
- **Tried:** pretrain JEPA → fine-tune at each fraction vs supervised baseline.
- **Result:** JEPA-pretrained **lost at every fraction** (test Dice — 10%: 0.24 vs 0.29, 50%: 0.59 vs 0.82, 75%: 0.59 vs 0.81, 100%: 0.64 vs 0.82). The opposite of the hypothesis. Pretraining *hurt*.
- **Takeaway:** something is wrong with either the pretraining objective or the comparison. Investigate before scaling.

### 2026-07-12 · Step 2 — Diagnosis.
- **Found:**
  1. **Collapse-prone objective** — one encoder was its own prediction target (no stop-gradient / EMA), so JEPA loss could go to ~0 with uninformative features.
  2. **Reconstruction never trained the encoder** — the pixel decoder detached the encoder, wasting a signal that would ground features spatially.
  3. **Fine-tune handicap** — encoder learning rate hardcoded to `lr/10`; the pretrained model was undertrained (val Dice still rising at 100 epochs) while the baseline converged by ~20.
  4. **No data advantage** — pretraining and fine-tuning used the *same single recording*, so SSL had nothing extra to exploit.
- **Takeaway:** the negative result was mostly an artifact of a broken setup, not proof the hypothesis is false.

### 2026-07-12 · Step 3 — Fixes.
- **Changed:** added an **EMA target encoder** (stop-gradient, anti-collapse); made **reconstruction train the encoder** (`recon_coeff`); made the fine-tune encoder LR **configurable** (`finetune_encoder_lr_scale`, set to 1.0 for a fair init-only comparison); made Neurofinder discovery **recursive** so pretraining can use the whole dataset; made data loading **lazy/memory-bounded** (fixed an out-of-memory crash on the full ~20 GB set).

### 2026-07-12 · Step 4 — Performance rescope.
- **Problem:** on the full 20 GB set, pretraining ran at ~1.8 h/epoch → 100 epochs ≈ 178 h, would never finish inside a session and saved nothing until the end.
- **Changed:** `pretrain_clip_stride` 2→10 (non-overlapping + frame subsample → ~5× less disk I/O per epoch), `pretrain_epochs` 100→20, `finetune_epochs` 100→40; added a **rolling pretrain checkpoint** every 5 epochs for crash recovery.
- **Takeaway:** the run was I/O-bound (lazy loading re-reads TIFFs every epoch); overlapping clips were the main waste.

### Status
Re-running on Kaggle with all fixes. Awaiting whether pretraining now helps —
watch especially the **0.1 fraction**, where SSL should show its largest advantage.

---

## H2 — Cross-organism transfer

**Question (final form):** train two segmentation models on organism 1 (source),
test both on organism 2 (target); do JEPA-pretrained features transfer better than
supervised ones? Source = zebrafish (neurofinder.04.00), target = mouse (00.00).

### 2026-07-12 · Step 1 — Design mismatch found.
- **Found:** the code trained the segmentation on the *target* and fine-tuned there — it did **not** match the hypothesis (which is: train on source, test on target). It answered a different, weaker question.
- **Takeaway:** realign the code to the actual hypothesis before trusting any result.

### 2026-07-12 · Step 2 — Realign to train-on-source / test-on-target.
- **Changed:** `run_h2` now trains both models entirely on the source, then evaluates them on the target. Rebuilt the plots to fit (zero-shot target comparison + source→target drop). Removed the fine-tune-on-target machinery (convergence/budget plots) that belonged to the wrong design.

### 2026-07-12 · Step 3 — Zero-shot run. Flatline.
- **Tried:** strict zero-shot — train on zebrafish, apply to mouse with **no** target adaptation.
- **Result:** both models nailed the source (Dice ~0.96–0.97) but **both flatlined on the target** (Dice **0.04** JEPA vs **0.03** supervised). Drop ≈ 0.94. (mIoU showed ~0.42 for both, but that is just the background class — the models predict essentially *empty* masks on mouse, which the Dice≈0 confirms.)
- **Takeaway:** **strict zero-shot cross-organism transfer was too harsh** — zebrafish and mouse are too different (intensity, neuron size/density, scale), so a small model sees mouse as fully out-of-distribution and predicts nothing above threshold. Both models sit at the floor, so the experiment **can't discriminate** which representation transfers better (0.04 vs 0.03 is noise). Not a bug — the same pipeline gets 0.96 on source.

### 2026-07-12 · Step 4 — Diagnose the floor (threshold sweep, AUPRC, AdaBN).
- **Tried:** on the source-trained checkpoints, evaluated on mouse with a threshold sweep, threshold-free **AUPRC**, and **AdaBN** (recompute BatchNorm stats on unlabeled mouse). In-domain sanity: the same pipeline scores the JEPA model on zebrafish at Dice **0.96** / AUPRC **0.99** (chance 0.12), so the eval is correct.
- **Result (mouse):** **AUPRC ≈ chance for both** — JEPA **0.157**, supervised **0.151** vs chance **0.164**. AdaBN barely moved it (→0.158 / 0.164); best-threshold Dice only ~0.10. JEPA ≈ supervised.
- **Takeaway:** the floor is **real, not an artifact**. The cheap fixes (thresholding, BatchNorm recalibration) recover nothing — the source-trained models produce essentially **noise** on mouse (their output carries no information about mouse neuron locations). Neither representation transfers zero-shot. Script: `scripts/h2_transfer_diagnostics.py`.

### 2026-07-12 · Step 5 — Planned: linear-probe on target ("minimal changes").
- **Why still worth it:** the Step-4 diagnostic tested the full model (encoder + *zebrafish* head). AUPRC-at-chance shows the zebrafish head can't decode mouse — it does **not** prove the *encoder features* are useless. A linear probe (**freeze encoder, train only a fresh head on a small slice of mouse labels**) tests exactly whether the encoder features transfer, and whether JEPA's are more organism-general than supervised. If the probe also floors, zebrafish→mouse is too far a transfer for this model → escalate to a closer pair (mouse→mouse), augmentation, or multi-organism pretraining.

### Status
Zero-shot floors (confirmed real). Next: build + run the target linear-probe.

---

## H3 — Temporal representation stability

**Question:** are learned embeddings more stable across time for the same neuron than
for different neurons? Compares within- vs between-neuron cosine similarity across
pretrained / supervised / random encoders. Post-hoc — needs H1 checkpoints first.

### Status
Not yet run this cycle. Blocked on a good H1 pretrained checkpoint from the re-run.

