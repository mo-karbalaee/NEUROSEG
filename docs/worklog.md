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

### 2026-07-12 · Step 5 — Full H1 run (Kaggle v8): clean pipeline & strong baseline, but SSL still loses.
- **Ran:** the complete H1 protocol — 20-epoch JEPA pretrain → finetune vs from-scratch supervised baseline at labeled fractions 0.1 / 0.5 / 0.75 / 1.0, on Neurofinder mouse. Output in `output/H1.v8/` (all 8 checkpoints + curves + `h1_dice_comparison.png` / `h1_miou_comparison.png`). Everything ran end-to-end and produced an interpretable performance-vs-labeled-fraction curve.
- **Final test Dice (pretrained→finetune vs supervised_baseline):**

| frac | pretrained | baseline | Δ (pre−base) |
|------|-----------|----------|--------------|
| 0.10 | 0.2225 | **0.3240** | **−0.10** |
| 0.50 | 0.6717 | **0.7427** | −0.07 |
| 0.75 | 0.6771 | **0.7615** | −0.08 |
| 1.00 | 0.7366 | **0.7812** | −0.04 |

  mIoU follows the same order (baseline higher at all four fractions: 0.523/0.753/0.780/0.792 vs 0.485/0.700/0.713/0.756).
- **What's genuinely good:** absolute quality is now strong — Dice ~0.74–0.78 at full labels on real mouse data (earlier H1 runs were near-collapse). Both arms rise monotonically with labeled fraction; the full protocol (pretrain + 4 fractions × 2 arms + plots) is complete and correct. The **machinery** is done.
- **But the hypothesis is contradicted, not supported:** H1 predicts SSL should help most at **low** fractions. Instead the pretrained model **loses to from-scratch at every fraction, by the widest margin at 0.1** (−0.10 Dice) — the opposite of the predicted pattern.
- **Root cause — pretraining diverged on validation (overfit):** JEPA train loss fell 2.59 → 0.30 over 20 epochs, but **val_jepa_loss rose 3.13 → 5.96** (spiking to 16.99 at epoch 14); val_recon stayed flat ~0.004 throughout. The predictive/VICReg objective memorized the small pretraining pool rather than learning a transferable representation, so initializing finetuning from that encoder is *worse* than random init. Same overfitting signature as H2.v3 (H2 Step 11), now confirmed to directly damage H1 transfer.
- **Fix is on the pretraining side, not finetuning:** more/cleaner unlabeled data (or less aggressive `pretrain_clip_stride` for more distinct clips), **save best-val checkpoint instead of last**, early-stop on val_jepa_loss, and stronger anti-collapse. The finetune/baseline halves are working correctly and need no change.
- **The headline positive — this is the semi-supervised paradigm, and it holds:** JEPA pretrains on the **entire Neurofinder dataset unlabeled** (`_make_unlabeled_dataset` → `NeurofinderDataset(labeled=False)`, `h1_trainer.py:108`); the labeled *fraction* gates **only the supervised finetuning** (`labeled_fraction`, `h1_trainer.py:127`), not the imagery the encoder sees. So the x-axis is not "how much data" but "**how many labels we paid for**" — and unlabeled calcium video is abundant/free while hand-drawn soma labels are the scarce, expensive resource. The label-efficiency curve then says the useful thing: **~95% of full-label Dice is reached with half the labels, ~98% with three-quarters** (baseline, as % of its own 0.781 @ 100%: 10%→41%, 50%→95%, 75%→98%). The last half of the labels buys only ~5% Dice. That is a real, defensible result — you can roughly halve the labeling budget at ~5% quality cost — and it stands whether or not SSL beats scratch. (Honest bound: "small fraction" here means **~50%**, not 10% — at 10% both arms are on the cliff, 0.32/0.22 Dice.)

### 2026-07-12 · Step 6 — INVALIDATION: the train/test split is broken — SSL + eval touched the same recording. v8 numbers do not count.
- **The failure:** in v8 the JEPA self-supervised pretraining ran on `--data = .../neuroseg-labeled` (the parent folder). `find_neurofinder_dirs` is recursive, so pretraining swept up **every** recording under it — **including `neurofinder.00.00`**. But 00.00 is exactly the recording we then **finetuned and tested on** (`--labeled-data .../neurofinder.00.00`). **So the self-supervised step saw the evaluation recording.** The pretraining step is *not supposed to see* the data we report test numbers on — it did. That is a leakage failure, full stop.
- **Compounded by a second leak (within-recording, static mask):** even ignoring pretraining, finetune+baseline both train and test on time-windows of the **single** 00.00 movie, and the segmentation target is a **static per-recording mask** (identical for every clip, `dataset.py:417`). `random_split` gives disjoint *clips* but they are windows of the same movie with the same fixed answer → the model memorizes 00.00's neuron layout and is tested on the same layout. Not a generalization test.
- **Verdict — v8 results are invalid and retracted:** the absolute Dice (~0.78), the label-efficiency curve, and the "halve the labels at ~5% cost" framing are all **inflated by leakage and do not count**. The earlier "headline positive" bullet above is **withdrawn** on these grounds. (The only thing that partially survives is the *relative* pretrained-vs-baseline gap at a fixed fraction, since both share the same leaky split — but with the SSL step also contaminated, even that is not worth reporting.)
- **What correct looks like:** (1) the JEPA pretraining pool must **exclude** the recording(s) used for downstream test; (2) evaluation must be **cross-recording** — finetune on one/some recordings, test on a **held-out different recording** (Neurofinder is built for this) — never within a single recording with a static mask. Both must hold at once for any H1 number to mean anything.
- **Status of the fix:** not yet implemented. Requires reworking `run_h1`/`finetune` to split by *recording* (pretrain-set, finetune-set, test-set are disjoint recordings) and updating the notebook to pass a multi-recording layout. Until then, **no H1 metric should be cited.**

### 2026-07-12 · Step 7 — Leakage fix implemented (commit 992de33). The SSL step no longer sees the test recording.
- **Design (agreed with supervisor):** (1) **JEPA pretrains on the entire dataset EXCEPT `00.00`** — 00.00 is held out of self-supervised training entirely; (2) **supervised + semi-supervised finetuning operate only on `00.00`**; (3) within 00.00 a **fixed 80/20 split** — 20% is the held-out **test set, identical for every labeled fraction and both models**; (4) the **labeled fractions subsample only the 80% train pool**, never the test.
- **Implementation:**
  - `NeurofinderDataset` gained an `exclude_dirs` arg (resolved-path match, `dataset.py`); `_make_unlabeled_dataset`/`pretrain` thread it through. `run_h1` now passes `exclude=[labeled_data_dir]`, so pretraining drops the test recording automatically.
  - `finetune` rewritten: builds the **full** 00.00 dataset (no pre-subsample), then a **seed-fixed** permutation → fixed 20% test, fixed val (`val_split` of the remaining pool), and `train = fraction × train_pool`. Test/val are independent of `fraction`, so all four points are scored on **one identical held-out set**. Each run prints `test=… (fixed) val=… train=…/pool`.
- **Verified:** unit-checked that the 20% test set is byte-identical across fractions 0.1/0.5/0.75/1.0 with zero train∩test overlap (train scales 7→36→54→72 on a 100-clip toy); synthetic 2-recording check confirms `exclude_dirs` drops 00.00's clips from the pretrain pool while keeping the others; full suite 17/17 green.
- **Notebook:** **no change needed** — it already passes `--data .../neuroseg-labeled` (whole) + `--labeled-data .../neurofinder.00.00`; the fix auto-excludes 00.00 from pretraining. Just re-clone `main` and re-run.
- **Residual caveat (acknowledged, chosen):** eval is still **within-recording** (train/test are disjoint 80/20 time-windows of 00.00 against its static mask), so the test measures within-00.00 performance, not cross-recording generalization. This is the supervisor's chosen protocol and the **primary** leak (SSL seeing the test recording) is now closed. A stricter cross-recording eval remains a possible future tightening.
- **Next:** re-run on Kaggle (call it v9) and read the pretrained-vs-baseline curve on the now-clean fixed test set.

### 2026-07-12 · Step 8 — Inference was silently broken: the segmenter never normalized its input. Masks fixed (commit f91b0df).
- **Trigger:** ran the uploaded v8 checkpoints in **inference mode** on 60 frames of `neurofinder.00.00` to *see* actual segmentations. The overlays looked like garbage — masks flooding ~half the frame.
- **Proved it empirically (not by argument), against the real target mask:**
  - The training data pipeline min-max normalizes every clip to **[0,1]** (`_normalize`, `dataset.py:409`). The segmenter fed the model **raw uint16 pixel values** (0–thousands) with **no normalization** — a ~1000× input-range shift from what the model trained on.
  - Same model, same frames: **raw input → 45% foreground, Dice 0.22**; **normalized input → 14% foreground, Dice 0.76** (target mask FG at 128 = 16.4%). The model was never the problem; the segmenter was feeding it out-of-range input and it flooded the frame.
  - A second, smaller mismatch: training averages encoder features over a **5-frame window** (`enc_state.mean(dim=2)`, `h1_trainer.py:336`); inference used a **single frame**. Single-frame normalized Dice 0.586 vs 5-frame-window 0.754 → temporal aggregation is worth ~+0.17 and real (single-vs-window masks agreed only IoU 0.625).
- **Fix (all inference-side; training/eval untouched, so the v8 leakage retraction is unaffected):**
  - `segmenter._jepa_segment` now **min-max normalizes each frame** and **averages encoder features over a 5-frame window** (features cached once per frame → cheap). Matches the training input distribution.
  - `min_size` speckle filter was mis-scaled (used `img_size²` = 163 px, applied to 512² masks). Swept it: at native 512, any `min_size ≤ 32` keeps Dice ~0.587 while `262` tanks it to 0.463 — set to `(H·W)//8000` (≈32 px at 512). NB: this filter was **not** the cause of the flood (I had guessed it was; retracted) — the flood was purely the missing normalization.
  - **Resolution note:** the model works at 128 (Dice ~0.75 there); viewed/scored at the native 512 its upsampled masks are Dice ~0.58 — still 2.6× the broken 0.22. Both numbers are legitimate; 128 is the model's trained resolution.
- **Visualization overhaul (user request):** the visualizer now writes **one MP4 per stack** (`segmentation/<stack>.mp4`, 10 fps, frame counter burned in) instead of a folder of per-frame PNGs, and draws **per-neuron contour outlines** (`cv2.findContours`/`drawContours`) instead of filled 50%-opacity blobs. Result on f100: ~131 neuron outlines/frame, FG 12.6%, Dice 0.58 @ 512 — real segmentation, cleanly rendered.
- **Verified:** suite 17/17; f100 inference re-run end-to-end producing a valid, playable MP4.
- **Command to reproduce:** `python main.py --mode inference --data <folder-of-TIFF-stacks> --output <dir> --checkpoint output/H1.v8/jepa_h1_finetune_f100_bb1b8658.pt` (only the compound `finetune_*`/`supervised_*` checkpoints work — `pretrained_*` are encoder-only, no seg head).

### 2026-07-12 · Step 9 — CRITICAL: the ground-truth masks were TRANSPOSED. `_build_nf_mask` read regions.json coords as [x,y] instead of [y,x] (commit a1335f9).
- **Trigger:** wrote a standalone GT-visualization script (`scripts/visualize_ground_truth.py`); the supervisor asked "are you sure the GT is right — the regions are constant?" (Constant is *correct*: Neurofinder regions are static spatial footprints; only brightness changes over time.) Verifying alignment surfaced the real bug.
- **The bug:** `_build_nf_mask` did `xs=coords[:,0]; ys=coords[:,1]; mask[ys,xs]` — i.e. it treated each coordinate as **[x, y]**. Neurofinder `regions.json` coordinates are **[y, x]** (row, col). Result: **every mask was transposed** (reflected across the main diagonal).
- **Proved it empirically** against a 300-frame **max-projection** (every neuron that ever fires is bright): mean brightness *inside* the mask vs outside — **[y,x] → ratio 1.46** (mask sits on the cells), **[x,y] = old code → ratio 0.99** (no alignment, floats over background). Confirmed visually (contours on the projection): [y,x] lands on cells, [x,y] does not. Fix: swap to `ys=coords[:,0]; xs=coords[:,1]`. Post-fix ratio 1.46; suite 17/17.
- **Blast radius:** `_build_nf_mask` builds **every** Neurofinder label — H1 training/finetune, the H2 probe, and all Dice/mIoU. So **every H1/H2 model trained against transposed targets, and every metric was scored against transposed masks.** The v8 checkpoints predict *transposed* footprints — their segmentations do not sit on real neurons; **they must be retrained.**
- **Why the numbers didn't look broken (the trap):** the mask is **static per recording** and eval is **within** 00.00, so the model just memorizes whatever static map it is handed — transposed or not — and reproduces it. The transpose barely moved the (already leakage-inflated) within-recording Dice (~0.75), which is why v8 "looked fine" while the overlays floated off the cells. The bug is devastating for *real* use and any *cross-recording* transfer, not for the leaky within-recording score.
- **Three independent bugs now fixed, stacking for a clean v9:** (1) leakage — SSL no longer sees the test recording (`992de33`); (2) inference normalization — masks stop flooding (`f91b0df`); (3) mask transpose — labels sit on real neurons (`a1335f9`).

### 2026-07-12 · Step 10 — SSL representation upgrades: augmentation + wider latent + best-val checkpoint (commit 2993ad7).
- **Motivation (from the H3 collapse + H1 divergence):** the v8 JEPA overfit (val_jepa_loss 3.13→5.96) and H3 showed representation collapse (within≈between≈0.98). Code audit found the causes: **zero data augmentation** anywhere, an **8-dim latent** (`dstc=8`, too small to keep 330 neurons distinct), and the pretrain loop **saving the LAST epoch** (the worst, since val diverges). Our own H3 evidence backed diversity as the fix (cross-species encoder was 3× less collapsed than mouse-only).
- **Implemented:**
  1. **Augmentation pipeline** (`_augment_clip` + `AugmentedClips` wrapper, `dataset.py`): temporally-consistent random resized-crop / h+v flip / 90° rotation, plus photometric jitter and Gaussian noise. Applied **only to the training split** (wrapper wraps `train_set` after the split), so validation stays clean and best-val selection is honest. On/off via `pretrain_augment` (default true).
  2. **Wider latent:** `dstc` 8 → **32** in `config.yaml` (4× capacity per location; directly targets the H3 discriminability ceiling — supervised reached between=0.67 at dstc=8, so the geometry is learnable, the SSL encoder just lacked room).
  3. **Best-val checkpoint + early-stop:** `pretrain()` now tracks the best `val_jepa_loss`, keeps that encoder's weights in memory, restores them before the final save (so the saved checkpoint is the **best** epoch, not the last), and early-stops after `pretrain_patience` epochs without improvement (default 8; 0 disables). Rolling `save_latest_checkpoint` still runs for crash recovery.
- **Verified:** `_augment_clip` preserves shape/range and changes content; end-to-end synthetic pretrain shows `augment=True | dstc=32`, early-stop firing, and `restored best-val encoder from epoch N` before save; full `run_h1` (pretrain→finetune, dstc=32) produces all 4 summary rows; suite 17/17.
- **Not yet tuned (future):** the VICReg std/cov weights (`std_coeff=10, cov_coeff=100`) and EMA momentum (0.996) — collapse persisted despite them; worth rebalancing once the above land. Multi-source pretraining (Neurofinder + Drosophila + zebrafish) is the other big data lever.

### Status
**v8 fully RETRACTED** (leakage `992de33`, inference-normalization `f91b0df`, GT-mask transpose `a1335f9`) — all prior Neurofinder masks/metrics void, retrain required. **Plus SSL-quality upgrades for v9** (`2993ad7`): training-only augmentation, `dstc` 8→32, best-val checkpoint + early-stop — aimed squarely at the pretraining overfit/collapse seen in H1 and H3. Suite 17/17; GT-viz + inference-viz verified on real cells. **Next: Kaggle/HPC v9 with all five fixes → first H1 result whose masks correspond to real neurons AND whose SSL encoder has a fighting chance of not collapsing (re-check via H3).**

---

## H2 — Cross-organism transfer

**Question (final form):** train two segmentation models on organism 1 (source),
test both on organism 2 (target); do JEPA-pretrained features transfer better than
supervised ones? *(Originally framed as zebrafish (04.00) → mouse (00.00) — this was
WRONG; Neurofinder is all mouse. See Step 7. It was really mouse parietal cortex →
mouse barrel cortex.)*

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

### 2026-07-12 · Step 6 — Linear probe. Also floors — the *features* don't transfer.
- **Tried:** froze each source-trained encoder, trained a fresh seg head on 1/5/10% of mouse labels, evaluated on held-out mouse. In-domain sanity: the same probe on zebrafish recovers Dice **0.95–0.96**, so the probe method works.
- **Result (mouse):** **Dice 0.000 at every fraction, for both** JEPA and supervised. Not a training artifact — the in-domain probe is 0.95.
- **Takeaway:** the deepest form of the negative result. The zebrafish encoder features carry **no usable mouse-neuron information at all**, even given a fresh mouse-trained head — so it is not the head, not calibration, not BatchNorm, not a bug. The representation is fully organism-specific, for both SSL and supervised. **Zebrafish→mouse is too far a domain gap for this model**; there is no signal on this pair to distinguish the two representations. Script: `scripts/h2_linear_probe.py`.
- **Decision — pivot:** (a) test a **closer pair** (mouse→mouse) to reach a regime with signal and characterize transfer vs domain distance; and/or (b) **multi-organism JEPA pretraining** (pretrain on several organisms, probe a held-out one) — where SSL's cross-organism advantage should appear.

### 2026-07-12 · Step 7 — CRITICAL: Neurofinder is ALL MOUSE. Our organism assumption was wrong.
- **Finding:** the entire Neurofinder benchmark is **mouse** two-photon calcium imaging — **no zebrafish, no other species at all**. The series number encodes the contributing lab / brain region, NOT the organism:
  - `00`, `02` — Svoboda / Janelia — mouse **barrel (somatosensory) cortex**
  - `01` — Häusser / UCL — mouse **barrel cortex**
  - `03` — Losonczy / Columbia — mouse **hippocampus (CA1)**
  - `04` — Harvey / Harvard — mouse **posterior parietal cortex**
- **Impact:** the code's `_NF_ORGANISM_MAP` (`04 = zebrafish`, `10 = mouse.hippocampus`; note series `10` does not even exist) is **fabricated and wrong**. Every H2 figure/log that said "zebrafish" was mislabeled — the run we called "zebrafish → mouse" was actually **mouse parietal cortex (04.00) → mouse barrel cortex (00.00)**: a cross-region / cross-lab transfer *within one species*. And it still floored to Dice 0.
- **Consequence for H2:** with Neurofinder alone, H2 **cannot be cross-organism** — at best it is **cross-region / cross-lab within mouse**. The largest usable domain gap is **cortex → hippocampus** (train on 00/01/02/04, test on **03**). A genuine cross-species test needs a non-Neurofinder dataset.
- **Sources:** [Neurofinder README](https://github.com/codeneuro/neurofinder/blob/master/README.md) (series → lab); Peron et al., *Neuron* 2015 (00/02 mouse barrel cortex); Packer et al., *Nat. Methods* 2015 (01 mouse barrel cortex); Kaifosh et al., *Front. Neuroinform.* 2014 / Losonczy lab (03 mouse CA1); Driscoll/Harvey et al., *Cell* 2017 (04 mouse parietal cortex).
- **Takeaway / TODO:** re-scope H2 as cross-*region* (or bring in a real second species). Fix `_NF_ORGANISM_MAP` so figures stop printing "zebrafish."

### 2026-07-12 · Step 8 — Bring in a real non-mouse species for pretraining (Drosophila + zebrafish).
- **New data:** the supervisor provided a dataset (Kaggle: `mokarbalaee/neuroseg-drosophila-larvae`) — mostly **Drosophila larvae** calcium imaging, plus 2 **zebrafish** files (named with "ETL", i.e. Electrically Tunable Lens volumetric imaging). This is genuine **non-mouse** data, which finally makes a *true* cross-species transfer possible (Neurofinder alone can't — it is all mouse, Step 7).
- **Plan:** pretrain JEPA **unsupervised on the whole non-mouse pool (Drosophila + zebrafish, mixed)** → then test transfer to **mouse** (Neurofinder, labeled) via linear-probe / fine-tune, compared against mouse-from-scratch. This is the real H2 question: do cross-species SSL features transfer to a *held-out species* (mouse) better than supervised features? Pretraining needs only unlabeled video, so labels on the Drosophila side are not required; the mouse labels drive the downstream test.
- **Status:** inspecting the dataset's file format on Kaggle (folder structure; whether each file is a multi-frame TIFF stack `(T,H,W)`; resolution; grayscale) to decide which loader to use before wiring the pretraining run.

### 2026-07-12 · Step 9 — Dataset identified and characterized.
- **Dataset:** Kaggle `mokarbalaee/neuroseg-drosophila-larvae` — the supervisor's non-mouse calcium imaging. We inspected it with a script on Kaggle (folder tree + file magic bytes).
- **Contents (20 files):**
  - **Drosophila larvae** — 2 TIFF stacks (`Ca imaging good (1-290)` = 290 frames, 512×248; `Ca imaging bad (1-320)` = 320 frames, 241×512) + **16 CZI files** (~200 MB each, ~388 frames of 512×512), split into `Ca imaging low movement/` (8 files) and `Ca imaging strong movement/` (8 files).
  - **Zebrafish** — 2 ETL 6-plane TIFF stacks (`joe ETL 6plane_00001_stacked` = 500 frames; `joe ETL 6plane 5kframes_00002` = **5000 frames, 15.8 GB**). Each frame is 512×**3072** = **6 z-planes stacked side-by-side** → must be split into 6× 512×512.
- **The `.sec` mystery, solved:** the 16 `.sec` files are **Zeiss CZI** microscopy files (magic header `ZISRAWFILE`; the filenames also literally contain "czi" — a bulk-rename turned `X.czi` into `Xczi.sec`). CZI is Zeiss ZEN's native microscope format, readable in Python via `czifile` / `aicspylibczi` / `pylibCZIrw`.
- **Takeaway:** ~3.2 GB Drosophila (CZI) + ~17 GB zebrafish (ETL tif) of genuine **non-mouse** video — plenty for SSL pretraining. The loader must read both `.tif` and `.czi/.sec`, and split the 6-plane ETL stacks. This is the pretraining pool for the cross-species H2 (→ transfer to mouse).

### 2026-07-12 · Step 10 — Implemented cross-species H2 (pretrain non-mouse → probe mouse).
- **Built:** `VideoFolderDataset` (`dataset.py`) — reads TIFF **and** Zeiss CZI (`.czi`/`.sec`), resizes on load (bounded memory), and splits the ETL 6-plane 512×3072 stacks into 6× 512×512; memory-safe via `tifffile.memmap` for the 15.8 GB file. Rewrote `run_h2`: pretrain JEPA (unsupervised) on `--source-data` (Drosophila + zebrafish) → `probe_on_target` freezes the encoder and trains only a seg head on `probe_fraction` (0.1) of the **mouse** target, evaluates held-out mouse — **pretrained vs from-scratch** encoder. New plot `plot_h2_probe` (mouse Dice/mIoU bars). Added `czifile` dependency.
- **Why a probe, not strict zero-shot:** the Drosophila/zebrafish data has **no segmentation labels**, so a pretrained model has an encoder but no head → zero-shot *segmentation* is impossible. The linear probe (encoder frozen, tiny head trained on a mouse slice, tested on held-out mouse) is the honest transfer measure; the encoder never trains on mouse.
- **Verified:** the loader (multi-page TIFF + single-page + 6-plane split) and the full `run_h2` flow end-to-end on synthetic data; 17 smoke tests pass. **CZI (`.sec`) reading is not testable locally — must be verified on Kaggle** (czifile installs via `pip install -e .`).
- **Run:** H2 notebook → `--source-data <drosophila dataset> --target-data <neurofinder.00.00>`; both Kaggle datasets must be attached.

### 2026-07-12 · Step 11 — First cross-species run (Kaggle v3): pretraining worked, probe crashed.
- **Ran:** H2 on Kaggle — pretrain JEPA on Drosophila+zebrafish (5 GB cap → the 15.8 GB zebrafish file excluded) → probe on mouse (`neurofinder.00.00`). Output in `output/H2.v3/`.
- **What worked:** CZI (`.sec`) loaded cleanly (czifile installed, no errors), 20 pretrain epochs completed (~4 min/epoch, ~77 min), checkpoint saved (`jepa_pretrained_h2_56f2880d`).
- **What broke:** the probe stage crashed — `RuntimeError: Expected all tensors on the same device, cuda:0 and cpu` in `probe_on_target`: encoder features were on GPU but the target masks stayed on CPU. A device bug that only surfaces on GPU — the CPU-only local test missed it. **Fixed** (masks moved to the encoder's device; commit a5500e0).
- **Also flagged — pretraining overfits:** JEPA train loss fell 4.65 → 0.41, but **val JEPA loss diverged** from ~0.9 (epoch 13) up to 3.37 (epoch 19). The non-mouse pool (after the 5 GB cap + `pretrain_clip_stride=10`) is small, so the encoder overfits. Levers: include the big zebrafish file (raise the cap), lower the clip stride for more clips, or stop earlier / save best-val.
- **Next:** get the transfer number — probe the saved encoder on mouse (locally on the downloaded checkpoint, or re-run Kaggle with the fix): pretrained vs from-scratch.

### 2026-07-12 · Step 12 — Skip-pretraining flow + `--data`-required argparse bug (Kaggle v4).
- **Added `--pretrained-ckpt` reuse path:** to avoid re-running the 77-min pretrain, `run_h2` now accepts an already-trained JEPA checkpoint — it reads the architecture from the checkpoint's `.json` sidecar (`with_suffix(".json")`, so pass the `.pt`), skips pretraining, and goes straight to the pretrained-vs-from-scratch mouse probe. Requires only `--pretrained-ckpt` + `--target-data`; `--source-data` no longer needed. Checkpoint to reuse: `output/H2.v3/jepa_pretrained_h2_56f2880d.pt` (+ its `.json`). Notebook updated to this command (commit b1f5ded).
- **What broke (v4):** the run died at **argument parsing**, before any code — `main.py: error: the following arguments are required: --data`. `--data` was globally `required=True`, but H2 uses `--source-data`/`--target-data` and ignores `--data` entirely, so the parser rejected the skip-pretraining command outright.
- **Fixed** (commit bf28107): `--data` is now optional; it is required for H1/H3/inference (validated in `main()` with a clear message) but not for H2. `run()` passes `--output` as a harmless stand-in data_dir for H2, and `pipeline.run` guards `iterdir` so a missing data_dir can't crash. Also corrected the stale `--pretrained-ckpt` help text (it said "(H3)"; it is an H2 flag).
- **Next:** re-clone `main` on Kaggle and re-run the probe cell (no edits) — get the pretrained-vs-from-scratch mouse Dice/mIoU.

### 2026-07-12 · Step 13 — First probe numbers (Kaggle v5): degenerate — the probe collapsed, not the transfer.
- **Ran:** the fixed skip-pretraining flow — reused `jepa_pretrained_h2_56f2880d.pt`, probed on `neurofinder.00.00`. Output in `output/H2.v5/`. It completed (both rows logged), so Steps 11–12 fixes held.
- **Result (`output/H2.v5/logs/runs.csv`):** pretrained → **Dice 0.000261, mIoU 0.418286**; from_scratch → **Dice 0.000000, mIoU 0.418243**. The two are statistically identical.
- **What the numbers actually mean — NOT a 42% score:** Dice ≈ 0 means the head predicts **essentially zero foreground pixels**. mIoU 0.418 is the signature of an **all-background prediction**: mIoU = (IoU_bg + IoU_fg)/2; with an empty mask IoU_fg = 0 and IoU_bg = fraction-of-background ≈ 0.836, so mIoU = 0.836/2 = **0.418**. That value only encodes "background is ~84% of pixels" — it is an **artifact, not signal**. Pretrained's 0.000261 is a couple of lucky pixels above threshold.
- **Root cause — the probe head collapses, the encoder is not being tested:** the head trains with plain `nn.BCELoss()` (`h2_trainer.py:100`) on masks that are ~16% foreground. With no positive weighting, BCE's optimal constant output is the base rate (~0.16), which is **below the 0.5 threshold → predicts background everywhere**. Combined with a tiny training set (`probe_fraction=0.1` → ~8% of a small clip pool) and a 2-conv head, both the pretrained and the random-init encoder settle on the same degenerate constant. So the probe **cannot distinguish a good encoder from a random one** — it is a broken measuring instrument, not evidence about Drosophila→mouse transfer.
- **Why it was suspiciously fast (~6 s for both):** expected for this design and it **confirms the `--pretrained-ckpt` skip worked** — no 77-min pretrain, the frozen encoder's features are cached in a single forward pass over a small clip set, and the 50 "epochs" are just a 2-layer conv over a few dozen cached tensors on GPU (milliseconds). Not a crash.
- **Caveat on earlier results:** the Step-6 "Dice 0.000" linear-probe floor used the **same BCELoss**, so that conclusion is partly contaminated by the same collapse — the features may never have been given a fair probe.
- **Fix to make it a real test:** (1) replace BCELoss with a class-imbalance-robust loss — **soft-Dice** (or `BCEWithLogitsLoss(pos_weight=neg/pos)`) — the one change that stops the all-background collapse; (2) add an **in-domain sanity mode** (probe mouse→mouse): if it can't reach ~0.8–0.9 there, the probe is still broken and any cross-species number is meaningless; (3) log the **all-background baseline (0.418)** next to mIoU so this artifact is never mistaken for signal again.

### Status
Cross-species pipeline runs end-to-end; Step 11 (device) and Step 12 (`--data` argparse) bugs are fixed and the v5 run completed. **But the probe result is degenerate:** the BCELoss head collapses to an all-background prediction for both pretrained and from-scratch (Dice ≈ 0, mIoU 0.418 = background-only artifact), so the run says nothing about transfer. Next: fix the probe loss (soft-Dice / pos_weight) + add an in-domain sanity check before trusting any cross-species number.

---

## H3 — Temporal representation stability

**Question:** are learned embeddings more stable across time for the same neuron than
for different neurons? For each encoder, pool a per-neuron embedding (encoder features
averaged over that neuron's footprint) per frame, then compare **within-neuron** cosine
similarity (same neuron across frames — want HIGH, stable identity) vs **between-neuron**
similarity (different neurons — want LOW, discriminable). **Gap = within − between**; a
large gap = stable *and* discriminative representation. Compared across pretrained (SSL) /
supervised / random encoders. Post-hoc analysis over H1's encoders — no training.

### 2026-07-12 · Step 1 — Interim run on the existing H1.v8 checkpoints (no retrain).
- **Why now:** needed a presentable H3 result for an interim talk without spending GPU. H3 uses only the **encoder**, and it runs locally in seconds — so the existing v8 checkpoints suffice for a first pass. Crucially, the **pretrained encoder is self-supervised (never saw masks)**, so the transpose bug (H1 Step 9) never touched it, and H3 pools over the **now-fixed** footprints — so this analysis is meaningful even though v8's *supervised* head is void.
- **Made it runnable:** the original `_compute_similarity_gap` did per-pair `cosine_similarity` in Python nested loops (604 clips × 330 neurons → timed out at 2 min). Rewrote it **vectorized** (L2-normalize rows, `V @ V.T`, upper triangle) and added `h3_max_clips` (sample clips evenly across the recording; default 40) → full 3-way run in ~30 s. Added a `--supervised-ckpt` CLI flag (H3's supervised encoder previously had no flag, config-only). Commit 29da2b7; suite 17/17.
- **Command:** `python main.py --mode train --H3 --data data/neurofinder.00.00 --pretrained-ckpt output/H1.v8/jepa_pretrained_h1_4621ee45.pt --supervised-ckpt output/H1.v8/jepa_h1_supervised_f100_70085721.pt --output output/H3_interim --config config.yaml`
- **Result (00.00, 40 clips) — HONEST: does NOT support H3 yet:**
  | encoder | within | between | gap |
  |---------|--------|---------|-----|
  | pretrained (SSL) | 0.986 | 0.976 | **0.009** |
  | supervised | 0.882 | 0.666 | **0.216** |
  | random (no_pretrain) | 0.984 | 0.981 | **0.003** |
- **Interpretation:** H3 predicts the **SSL** encoder should have the largest gap. Instead the **supervised** encoder wins by ~20×, and the SSL gap (0.009) is essentially **random** (0.003). The SSL features are **near-collapsed** — within ≈ between ≈ 0.98, i.e. every neuron maps to almost the same vector, so the encoder can't tell neurons apart. This is the **same v8 pretraining failure** seen in H1 (overfit/diverged JEPA), now visible as representation collapse. The supervised encoder legitimately learned discriminative per-neuron features (between drops to 0.67).
- **Fairness note:** run on 00.00; the v8 pretrained encoder saw 00.00 during SSL (pre-leakage-fix) and the supervised encoder trained on it — so both saw the recording; the comparison isn't confounded by one having seen it and the other not.
- **For the interim talk:** the honest arc is *method built → preliminary measurement → SSL collapse identified (a known, already-fixed cause) → clean re-run pending*. NOT "SSL wins." Plot: `output/H3_interim/figures/h3_similarity.png` (within-vs-between bars + gap bars).

### 2026-07-12 · Step 2 — Added the H2 cross-species encoder to the comparison (4-way).
- **Idea:** also run H3 on the **H2 pretrained encoder** (JEPA self-supervised on **Drosophila + zebrafish**, `output/H2.v3/jepa_pretrained_h2_56f2880d.pt`, same arch henc32/dstc8/img128) — i.e. does a *cross-species* SSL encoder represent **mouse** neurons any better than the mouse-SSL one?
- **Made it possible:** added `h3_extra_encoders` config (list of `{mode, ckpt}`) so H3 evaluates arbitrary named encoders beyond the default pretrained/supervised/random; extended the plot palette past 3 modes. Commit f065d20; suite 17/17. `_load_encoder` reads each checkpoint's own `.json` sidecar for arch, so mixed checkpoints load correctly.
- **Result (00.00, 40 clips):**
  | encoder | within | between | gap |
  |---------|--------|---------|-----|
  | supervised | 0.882 | 0.666 | **0.216** |
  | **H2 cross-species SSL** (Drosophila+zebrafish) | 0.968 | 0.940 | **0.028** |
  | H1 mouse SSL | 0.986 | 0.976 | **0.009** |
  | random | 0.978 | 0.973 | **0.005** |
- **Interpretation:** supervised still wins by ~8× — H3's core claim (SSL > supervised) remains **unsupported**. **But** the cross-species encoder is the **least-collapsed SSL**: its gap (0.028) is **3× the mouse-SSL encoder** (0.009) and **5× random** (0.005). So the Drosophila+zebrafish pretraining retained *more* mouse-neuron discriminability than the mouse pretraining did (which diverged worst in v8). A modest but real signal that broader/more-diverse pretraining resists collapse better — a secondary observation for the interim talk, not a headline.

### Status
**H3 measurement tool built and runs (fast, local); 4-way comparison in hand; first result is an honest negative with one interesting nuance.** On v8 checkpoints the supervised encoder is stable+discriminative (gap 0.216) while all SSL encoders are near-collapsed — but the **cross-species** SSL (0.028) is 3× less collapsed than **mouse** SSL (0.009), both above random (0.005). Consistent with the broken v8 pretraining; the hypothesis gets its real test only after the **clean v9 H1 retrain** (leakage + normalization + transpose fixes). Then re-run this 4-way H3 on the v9 encoder. Optional: a second recording (e.g. 04.00) to show the pattern holds.

