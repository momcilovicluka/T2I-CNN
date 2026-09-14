# Professor-style review — `T2I CNN - Luka Momčilović 39d.25.pdf`
**Revalidation of the current export, 14 Sept 2026. Scope: §§1–5 + front matter (citations ignored).**

Reviewed file: `T2I CNN - Luka Momčilović 39d.25.pdf` (42 pp., exported 13 Sept 16:58, the same
export `Plan/professor-review.md` calls "revalidation 4 / fifth export"). This pass is a
**revalidation against the files on disk**, plus independent checks of every number and
setting the text asserts. Where an item was already reported in `Plan/professor-review.md`
it is marked *[known]*; where it is new it is marked *[NEW]*.

Method: every checkable claim in §§1–5 was verified against (a) the code at HEAD
(`802bb17`) — `run_all.py`, `src/train.py`, `src/preprocessing.py`, `src/gradcam.py`,
`src/ablation.py`, `src/t2i/*`, `src/baselines/*` — (b) the installed `TINTOlib` 1.3.1
source, and (c) the recorded results in `results/`. Shape/parameter claims were measured by
running the models, not read off documentation.

---

## 1. Status of the previously flagged items in this export

| ID | Subject | `professor-review.md` says | State in **this** PDF | Action |
|---|---|---|---|---|
| C1 | No results in §6 | "still has no results" | p. 38 has the heading `6.1 Breast Cancer Wisconsin` and **nothing under it** | Write §6 from the real grid (`Plan/seminar2-rad-nacrt.md` §6 is the source; see `results-validation-2026-09-14.md`) |
| C1b | TOC field error | "TOC field errors fixed ✅" | p. 2 **still prints** `8.1 PRILOZI ............ ERROR! BOOKMARK NOT DEFINED.` | Marked fixed but present in this export — re-check the export/update the field before printing |
| M3 | Grad-CAM layer | "listing now misquotes the code" | Still true, and now measurable — see **§2.1** (it is the one item where the code, not the text, is wrong) | **✅ APPLIED 14 Sept** — code switched to `layer2`, comments corrected, draft §5.11 aligned, figures regenerated (see §2.1) |
| m6 | F1-comparability caveat | "still undisclosed" | p. 37 still ends `Primarna metrika za rangiranje je F1.` with no semantics | Insert the caveat; the backfill it depends on has now been run (45/45 JSONs carry `f1_macro_all`) |
| m7 | MLP early-stop wording | "exact replacement given" | p. 27 unchanged: `MLP (slojevi 128-64, ReLU, rano zaustavljanje na 10% validacije)` | Apply the replacement — the code confirms the objection: `validation_fraction=0.1` is carved out of `X_train`, not the shared val split |
| m8 | `scikit-learn>=1.3.0` permits the unweighted MLP fallback | "change the bound" | `requirements.txt:23` still `scikit-learn>=1.3.0` | Change to `>=1.4`; then grep the Colab log for `[mlp] … without per-sample balancing` |
| m10 | `TINTOlib` unpinned | "pin it / fix Table 5.2" | `requirements.txt:37` still bare `TINTOlib` | `pip freeze` in the grid's Colab session and pin; correct Table 5.2 if ≠ 1.3.1 |
| m11 | Doubled period in §7 | "one doubled period left" | p. 39 still reads `…interpolacije).. Pored toga…` | Delete the extra `.` |
| m1 | Stray closing quote (TINTO) | "one stray quote left" | p. 13 still ends `…konvoluciona jezgra 3×3.“` | Delete the stray `“` |
| m12 | §7 draws no conclusion from this thesis | skeleton given | p. 39 unchanged — still literature-level, no number from the grid | Port the skeleton |
| — | Adult dual numbers (48.842/14 vs 45.222/104) | exact Table 5.1 fix given | Still present (p. 22 table vs p. 23 Table 5.1) | Apply the given note |
| — | `pomoću` dangling (p. 37), `distibucije` (p. 23), `distribution shifits` (p. 7) | listed as proofreading debris | All three still present, verbatim | Fix with them |
| — | ShallowCNN 618,178 params, `features[8]` valid, 45-matrix, split procedure, Listing 5.1, class ratios | "verified clean" | Independently re-verified — all correct | keep |

**So: none of the previous round's action items has been applied yet**, apart from the
items the review itself marked ✅. Treat §2 below as additions to that same list.

---

## 2. New findings (not in `professor-review.md`)

### 2.1 `[NEW — the only code error; FIXED 14 Sept]` The Grad-CAM listing, the prose, and the code disagreed, and the code was the wrong one

**Where.** PDF p. 36, prose + Listing 5.10; `src/gradcam.py:23–40`; the figures
`results/figures/ch4_gradcam_*.png`.

**Measured facts** (ResNet-18, 32×32 input, run on this machine):

```
stem → 8×8   layer1 → 8×8   layer2 → 4×4   layer3 → 2×2   layer4 → 1×1
ShallowCNN features[8] → (1, 128, 4, 4)   Conv2d 3×3, 128 channels
```

**The problem.** The PDF prose and its Listing 5.10 say `layer2` (4×4) — which is the
arithmetically **correct and better** choice. The code said:

```python
# Use layer3 instead of layer4 — layer4 produces 2x2 maps on 32x32
# layer3 produces 4x4 maps which are still small but better
return model.backbone.layer3[-1].conv2
```

Every claim in those two comment lines is false for a 32×32 input (layer4 is **1×1**, not
2×2; layer3 is **2×2**, not 4×4), and `Plan/seminar2-rad-nacrt.md` §5.11 carried the same wrong
arithmetic plus the same listing — i.e. the PDF and the draft disagreed with the code and with
each other.

**Impact — corrected after reading the figure generator.** `src/visualize.py`
(`plot_gradcam_grid`) is the only production caller of `generate_gradcam`, and it hard-codes
`arch = 'shallow'`; the ResNet branch was therefore **not exercised by any figure**, and the
`ch4_gradcam_*.png` panels were always ShallowCNN `features[8]` (4×4) maps. So this was a
latent defect — wrong comments in a dead branch, plus a draft code-listing that did not match
the code — not corrupted figures. It becomes real the moment anyone runs Grad-CAM on a ResNet
cell, and it is the kind of listing a supervisor may diff against the repository.

**Applied 14 Sept 2026:**

1. `src/gradcam.py`: `layer3[-1].conv2` → `layer2[-1].conv2` (verified: the returned module is
   `layer2.1.conv2`, 128 channels, 4×4 at 32×32).
2. All comments corrected to the measured sizes (module docstring, `get_target_layer`
   docstring, inline comment).
3. `Plan/seminar2-rad-nacrt.md` §5.11: prose now gives the measured 8×8/4×4/2×2/1×1 table and
   says why `layer2` is chosen; Listing 5.10 mirrors the new code exactly.
4. The three `ch4_gradcam_*.png` regenerated from the stored `*_model.pt` (content unchanged by
   design — the ShallowCNN path was never affected; this only re-confirms the generator runs
   against the edited file).
5. The PDF needs no text change: its prose and listing already describe `layer2`/4×4, which is
   now what the code does.

**Second, separate mismatch found while fixing this (in the draft, not the PDF).** The §6.6
Grad-CAM caption claimed *"Grad-CAM se računa u odnosu na stvarnu klasu, ne na predikciju"*,
but the generator passes `pred` (`src/visualize.py:1790`), i.e. it explains the model's own
decision. Caption corrected to the predicted class; if the true class is actually wanted, the
call site must change to `true_label` and the figures regenerated — that is a scientific choice,
not a bug fix, and both the PDF prose (*"prilikom predviđanja određene klase"*) and standard
Grad-CAM practice favour explaining the prediction.

### 2.2 `[NEW]` The baselines' balanced class weights are never disclosed — and the text implies the opposite

**Where.** PDF p. 27: *"Svi su trenirani podrazumevanim/referentnim konfiguracijama bez
podešavanja… Namerno odsustvo podešavanja mora se imati u vidu pri čitanju razlika."*

**What the code does.**

* `src/baselines/rf.py:25` — `class_weight='balanced'` (a **non-default** sklearn setting).
* `src/baselines/xgboost_model.py:20-24` — per-sample `compute_class_weight('balanced')` weights.
* `src/baselines/mlp.py:26-33` — same balanced `sample_weight`.

**Why it matters.** This is the fairness argument for the whole CNN-vs-baseline comparison:
without it, baselines would be denied the imbalance remedy the CNN loss receives
(`compute_class_weight('balanced')` in the training config on the same page). As written, the
thesis tells the reader the baselines are "defaults without tuning", which is the opposite of
what was done, and a supervisor who asks *"were the baselines handicapped on the 75:25 set?"*
finds no answer. (The MLP's *internal* 10% split is covered by m7; the weighting itself is not.)

**Exact replacement** for the first sentence of that paragraph:

> *„Kao donja granica uporedivosti koriste se: Random Forest (100 stabala, bez ograničenja
> dubine), XGBoost (100 stabala, dubina 6, stopa 0,1) i MLP (slojevi 128-64, ReLU,
> `max_iter=500`, rano zaustavljanje na internih 10% trening redova). Svi dobijaju
> balansirane težine klasa (RF `class_weight='balanced'`, XGBoost i MLP preko per-sample
> `sample_weight`) — isto sredstvo protiv disbalansa koje dobija i CNN gubitak — ali im nijedan
> hiperparametar nije podešavan; namerno odsustvo podešavanja mora se imati u vidu pri čitanju
> razlika.“*

### 2.3 `[NEW]` The LP-FT ablation's "direct fine-tuning" arm is a separate run whose number differs from the main table

**Where.** PDF p. 36, the LP-FT paragraph; `src/ablation.py:466-490`; `results/ablation_lpft_*.json`.

**Facts.** The ablation trains its own `direct_ft` arm and records it separately:

| dataset | `direct_ft` F1 (ablation) | main-table `resnet` cell | LP-FT F1 | Δ |
|---|---|---|---|---|
| breast_cancer | **98.63 %** (37 ep.) | **97.22 %** | 91.04 % | −7.59 pp |
| dry_bean | 93.74 % | 93.79 % | 92.81 % | −0.93 pp |
| adult_income | 66.44 % | 66.19 % | 65.86 % | −0.59 pp |

The code itself flags this (`ablation.py`, "the direct-FT arm is a SEPARATE run from the
main-table resnet cell, not a reproduction of it (same config gave 98.63 here vs 97.22 in the
main run)"). Second issue: the two arms do **not** run at one learning rate — direct FT runs at
`ARCH_LR['resnet']` = 1e-3, while LP-FT is `lr=1e-3` for the 10 frozen epochs then
`lr_ft=1e-4` for the 40 fine-tuning epochs. The PDF mentions only the direct arm's rate.

**Why it matters.** As written, a reader expects the ablation to reproduce the main-table cell
and will find 98.63 ≠ 97.22 and 1.41 pp of unexplained difference; and "LP-FT loses 7.59 pp"
is then partly a learning-rate effect, not purely a representation-strategy effect.

**What to change and how.** Add two sentences at the end of that paragraph, and use the
ablation's own `direct_ft` value in §6.6 — never the main-table value:

> *„Direktno fino podešavanje u ovoj ablaciji je zaseban trening iste konfiguracije (isti seed),
> pa se njegov F1 može razlikovati od odgovarajuće ćelije glavne tabele; u poređenju se koristi
> vrednost iz same ablacije. Kraci se razlikuju i po stopi učenja: direktno fino podešavanje
> koristi `ARCH_LR` (1e-3) kroz sve epohe, dok LP-FT koristi 1e-3 u zamrznutoj fazi i 1e-4 u
> fazi finog podešavanja.“*

### 2.4 `[NEW]` The feature-ordering ablation cannot be informative for three of the four methods — the text presents it as if it can

**Where.** PDF p. 36, third ablation paragraph: *"Ukoliko se F1 značajno razlikuje između ovih
varijanti, može se zaključiti da način prostornog aranžiranja atributa ima merljiv uticaj na
rezultat modela."*

**Why it matters.** For DeepInsight, TINTO and IGTD the pixel layout is derived from
inter-feature relations (PCA loadings / rank distances), so **the column order of the input
cannot change the layout** — all four orderings are guaranteed to give identical results, and
they do (the re-run grid confirms 96.45 / 93.37 / 66.53 % identically for DeepInsight). Only
the naive method maps columns straight to pixels, so only there is the test meaningful. Written
as it stands, §5.3 promises a test that is vacuous for 3 of 4 methods, and §6 would then read
as a failed experiment instead of a statement about which methods even have a spatial layout
to defend.

**Exact addition** after that sentence:

> *„Ova ablacija ima smisla samo za metode kod kojih redosled kolona određuje raspored piksela;
> kod DeepInsight-a, TINTO-a i IGTD-a raspored se izvodi iz odnosa među atributima, pa sva
> četiri poretka daju identičnu sliku — za njih je ovo kontrolna provera invarijantnosti, a ne
> test uticaja rasporeda.“*

### 2.5 `[NEW]` The 3-channel control is missing from §5.3, although §5.3 states exactly the confound it removes

**Where.** PDF p. 32: *"Posledica je poređenje pretreniranog i od-nule ResNet-a nije čisto
poređenje efekta pretreniranosti već se razlikuje i ulazni domen."* — and the paragraph ends there.

**Facts.** The project has that control: `run_all.py` `--scratch-3ch` (`SCRATCH_3CH`), recorded in
`results/scratch3ch/` (3 runs, `scratch_input: "3ch-imagenet"`). It is also elided from the PDF's
Listing 5.7, although `create_cnn_model` contains the `if SCRATCH_3CH:` branch.

**What to change and how.** Convert the "consequence" into "consequence + how it is removed":
add after that sentence —

> *„Zato je izvedena i kontrola sa istim 3-kanalnim ImageNet-normalizovanim ulazom i bez
> pretreniranih težina (`--scratch-3ch`); time se uticaj ulaznog domena i uticaj pretreniranosti
> razdvajaju, a rezultat kontrole se navodi u §6.6.“*

and, in §5's experiment-matrix sentence, note that the 45-cell matrix is the main grid only
(the ablations, the control and the seed sweep are additional runs).

### 2.6 `[NEW]` Truncated sentence in §4.2 (Precision)

**Where.** PDF p. 19: *"…i ukupnog broja svih primera koje je model označio kao [38]."*

The sentence stops mid-clause ("as …"), leaving the definition incomplete.

**Fix:** insert the missing words: *„…koje je model označio kao **pozitivne** [38].“*

### 2.7 `[NEW]` A code fragment leaked into the body text at the top of page 29

**Where.** PDF p. 29 begins with:

> `images.min(), images.max())). Pošto se opsezi validacije/testa razlikuju od trening opsega…`

That is a wrapped line of the **previous** page's `np.clip(resized,` statement; the paragraph
then closes with a broken sentence: *"…i primenjuju na sve podskupove transformacija je tada
ista funkcija za trening, validaciju i test."*

**Fix:** delete the orphan `images.min(), images.max())).` and split the sentence:

> *„Ispravka je bila da se statistike izračunaju jednom, u `fit()`, na trening skupu, i primene
> na sve podskupove; transformacija je tada ista funkcija za trening, validaciju i test.“*

### 2.8 `[NEW]` Typo: `evalucaiju`

**Where.** PDF p. 5: *"Takva benchmark struktura omogućava evalucaiju T2I pristupa…"* → `evaluaciju`.

### 2.9 `[NEW]` Two metric definitions in §4.2 are imprecise in ways §6 will expose

**Where.** PDF p. 20.

* *"Vrednost AUC se kreće u rasponu od 0.5 do 1.0."* — AUC is defined on [0, 1]; 0.5 is the
  chance level, below which a model is worse than random. Fix: *„…u rasponu od 0 do 1, pri čemu
  0,5 odgovara slučajnom pogađanju.“*
* F1 is defined without saying *which* F1. §6 reports the **positive-class** F1 for
  breast_cancer/adult_income and **macro-F1** for dry_bean (7 classes), so §4.2 (or the §6
  preamble) must state the convention once — this is the same issue as *[known]* m6, so solve
  both with one insertion at the end of §5.3 (*"Primarna metrika…"*).

### 2.10 `[NEW]` §5.2 calls figures 5.4/5.5 "class distributions" — they are attribute distributions

**Where.** PDF p. 23: *"…na slikama 5.4 i 5.5 mogu videti numeričke i kategoričke distibucije
klasa skupa Adult Income."* The figure captions on p. 25/26 read *"Distribucija numeričkih
atributa…"* / *"Distribucija kategoričkih atributa…"*. Fix the sentence to *„…distribucije
numeričkih i kategoričkih **atributa** skupa Adult Income“* (and the `distibucije` → `distribucije`
typo listed in the previous review).

### 2.11 `[NEW, minor]` Capacity rounded two ways

§1 (p. 3) says *"ShallowCNN 0.62M, ResNet-18 11.2M"*; Table 5.3 (p. 27) says *"620K"* and *"11M"*.
Measured: **618,178** (0.618M) and **11,177,538** (11.18M). Use one pairing everywhere —
`0,62M / 11,2M` in both places is consistent with the measurements.

### 2.12 `[NEW — ties to today's results validation]` "T4 GPU" and "one machine" are stronger than the artefacts support

**Where.** PDF p. 37: *"Eksperimenti izvršeni na T4 GPU (Google Colab)…"*, and §6's planned
preamble ("jedna verzija koda na jednoj mašini").

**Why it matters now.** The stored JSONs record only `device: "cuda"` — no GPU model, no library
versions — so "T4" cannot be checked from the artefacts. And today's validation of the finished
run (`Plan/results-validation-2026-09-14.md` §2) shows the identical configuration
(seed 42 / split 42, `adult_income/naive/resnet`) producing **two** deterministic outcomes across
environments: 57.58 % six times and 69.08 % three times, identical to 15 digits *within* each
group and differing from the first validation epoch *between* groups. `src/train.py:52-53` sets
`cudnn.deterministic=True` / `benchmark=False`, so within one instance the pipeline is
reproducible — the residual variance is environmental.

**What to change and how.**

1. Record it: add the GPU name and the frozen versions (`torch`, `torchvision`, `timm`,
   `TINTOlib`, `scikit-learn`, `xgboost`) to the results JSON or to a one-page appendix, and
   attribute each run.
2. Soften the claim to what is verifiable: *„Eksperimenti su izvršeni na GPU instancama Google
   Colab okruženja (CUDA); verzije biblioteka navedene su u prilogu.“*
3. In §6's preamble, drop "na jednoj mašini" or qualify it, and keep §6.7's treatment of the
   flagged cell as an environment-sensitive artefact (never as a one-off).

---

## 3. Independently verified as correct (no change needed)

These were measured/checked directly, so they can be defended as-is:

| Claim in §§1–5 | Verified against |
|---|---|
| 45-cell matrix: 4 T2I × 3 datasets × 3 architectures = 36, + 3 baselines × 3 = 9 | `run_all.py`, `results/` |
| Splits 70/10/20 with val = 10 % of the whole, stratified, seed 42 | `preprocess()`; Listing 5.1 matches the code **line for line** |
| Split sizes actually used: 398/57/114 · 9527/1361/2723 · 31654/4523/9045 (test = 20 %) | run log |
| Dataset facts: 569/13 611/45 222 rows; 30/16/104 features; 63:37; 6,8:1; 75:25; Adult UCI train+test merged; `?` rows dropped before one-hot | `src/preprocessing.py`, data |
| Training config: Adam, wd 1e-4, lr 1e-3 for all three architectures, label smoothing 0.1, class weights `balanced`, patience 15, RLROP 0.5/patience 5, 50 epochs, batch 32, best-val-loss checkpoint, `mode='min'` | `run_all.py:65-70`, `src/train.py:103-130,196-223` |
| Early-stop arithmetic: patience 15 (not 10) argued against the scheduler's patience 5 | code |
| T2I parameters in Table 5.2 | `src/t2i/*`: IGTD `Pearson`/`Euclidean`/`squared`/1000/50 ✅; TINTO `blur=True, amplification=3.14, distance=2, steps=4, times=4`, `random_seed=42` ✅; DeepInsight PCA + collisions averaged (`group_method=avg_option` **is** TINTOlib's default) ✅; naive `ceil(sqrt(d))` + bicubic + **train**-only min-max ✅ |
| 32×32 monochrome canvas for all cells; pretrained arm gets 3-channel ImageNet-normalised RGB, from-scratch arm 1-channel grey | `DATASET_CONFIG`, `create_cnn_model`, Listing 5.7 faithful to code |
| Listings 5.1, 5.7, 5.9, 5.11 are faithful transcriptions of the code (only 5.10 is not — §2.1) | code |
| Parameter counts 618,178 / 11,177,538; `features[8]` is the 128-channel conv with 4×4 output | measured by forward pass |
| Seed sweep design (5 seeds, same training path, separate storage, mean ± sd, main table untouched); thresholds 0.02/0.01 are display-only, not statistical | `scripts/seed_sweep.py`, `src/ablation.py` |

---

## 4. Minimal edit list, in the order I would do it

1. **Code — ✅ done 14 Sept:** `src/gradcam.py` → `layer2` + comments (§2.1);
   `Plan/seminar2-rad-nacrt.md` §5.11 prose + Listing 5.10 aligned; the three Grad-CAM figures
   regenerated locally.
2. **Text, protocol fidelity (20 min):** balanced class weights for baselines (§2.2), MLP
   wording (m7), LP-FT arms and learning rates (§2.3), ordering-ablation scope (§2.4),
   3-channel control (§2.5).
3. **Text, defects (20 min):** TOC `ERROR! BOOKMARK NOT DEFINED` (C1b), truncated Precision
   sentence (§2.6), p. 29 orphan fragment + broken sentence (§2.7), `evalucaiju` (§2.8),
   AUC range (§2.9), figures 5.4/5.5 mislabel (§2.10), doubled period (m11), stray quote (m1),
   `pomoću` / `distibucije` / `shifits` (known), capacity rounding (§2.11).
4. **Environment honesty (10 min + 1 run):** record GPU + library versions, soften the "T4 /
   one machine" claim (§2.12).
5. **Before §6 is written:** the F1-convention caveat (m6/§2.9), the ablation caveats that §6.6
   must not over-claim (§2.3, §2.4), and the §7 rewrite (m12).

Nothing above requires retraining. The only figure regeneration is the Grad-CAM set, and it uses
the stored `*_model.pt` weights.

---

## 5. Out of scope (citations), listed only where the document breaks

Ignored as instructed, but three are structural rather than "missing":

* §5.1/§5.2 still contain literal placeholder strings — `[cite: 3, 4, 5]`, `[cite: 5, 6]`,
  `[cite: 7]` … — i.e. the Zotero field did not render.
* Reference `[50]` is cited in §4.1, but the reference list ends at `[43]`.
* In the §5.2 strategy table, S-IGTD is supported by `[12]`, which is the BIE paper; the
  dataset rows point to unrelated works (Breast Cancer → `[8]` EEG seizure study, Adult Income →
  `[10]` Multi-representation DeepInsight).
