# Professor-style review: T2I CNN — Luka Momčilović 39d/25 (PDF, 42 pp.)

Scope: the whole document (§§1–8). Citations ignored per request, except
dangling references that break the document structure.

Method: every checkable claim was verified against (a) `results/*.json`
(the 4 Sept 2026 grid: 36 CNN + 9 baselines), (b) the code in `src/` +
`run_all.py` at HEAD (`802bb17`), and (c) the corrected draft
`Plan/seminar2-rad-nacrt.md` for the numbers the rewritten §6 must use.

Version note (revalidation 4, 13 Sept 2026, fifth export, still 42
pp.): full fresh pass with emphasis on verifying the exact-text fixes
as pasted. Confirmed applied: m1 (blur sentence), m4 (both edits),
m11 (bicubic), M4 grammar, M5 (MLP sentence). Corrections to my own
prior flags: the last *„mrežnog saobraćaja“* occurrence is legitimate
BIE literature context (§2, network-traffic applications of *other*
authors' method) — M5 is now fully closed, and I was wrong to keep it
open. TOC page numbers spot-checked (§3→10, §4→15, §5→21, §6→38,
§7→39, §8→40: all correct). New: TOC lists 8.1 PRILOZI (p. 42) but the
body has no §8.1 — references end at [43]. Sharpened: m8
(`requirements.txt` actually pins `scikit-learn>=1.3.0`, which
explicitly permits the broken case).

Severity: **CRITICAL** = fails a defense as-is · **MAJOR** = a professor
will flag it · **MODERATE** = fix before submission · **MINOR** = polish.

---

## CRITICAL

### C1. §6 still has no results; TOC field errors fixed ✅

The intrusion content is gone (right call) and the newest export no
longer prints `ERROR! BOOKMARK NOT DEFINED` — dead 6.2/6.3 entries
removed, verified. What remains is p. 38: the heading *„6.1 Breast
Cancer Wisconsin“* followed by a blank page, then §7. No tables, no
numbers, no discussion.

**How to fix:** write §6 from the real grid (corrected key numbers in
the Appendix — *not* the −11.40 pp figures from the first review).

---

## MAJOR

### M1. ✅ VERIFIED FIXED — test set 20%, not 10%

p. 37 now reads *„20% svakog skupa test, 10% validacija“*, consistent
with p. 21, Listing 5.1 and `test_size=0.2`. No action.

### M2. ✅ VERIFIED FIXED — DeepInsight now PCA

p. 12 reads *„koristi postupak PCA“*; the confused TINTO aside is gone;
consistent with code and Table 5.2. No action.

### M3. Grad-CAM layers: listing now misquotes the code (regression in newest export)

Situation as of the newest export, verified against both artifacts:

- Prose (p. 35): **layer2 → 4×4**. Arithmetically correct for
  unmodified torchvision ResNet-18 strides on 32×32 (conv1→16,
  maxpool→8, layer1→8, **layer2→4×4, layer3→2×2, layer4→1×1**;
  strides preserved per `src/models/resnet_wrapper.py:49-57`).
- Listing 5.10: edited to `model.backbone.layer2[-1].conv2` — but
  **`src/gradcam.py:38` still returns `layer3`**, and its comments
  (lines 9-11, 31, 36-37) still assert the wrong sizes (*„layer4 …
  2x2“*, *„layer3 … 4x4“*). Working tree otherwise clean, so no code
  change accompanied the listing edit.
- Consequence: the thesis documents code that does not exist, and the
  generated `ch4_gradcam_*.png` figures still come from **layer3
  (2×2)** maps. A professor diffing the listing against the repo catches
  this immediately, and it reads worse than a plain error — like
  evidence edited without rerunning.

**How to fix (all three, atomically — exact code change):**
in `src/gradcam.py`, change `return model.backbone.layer3[-1].conv2`
to `return model.backbone.layer2[-1].conv2`; fix the comments to
*„layer4 produces 1x1 maps on 32x32“* / *„layer2 produces 4x4 maps“*
(the current comments are wrong about both layers); rerun the Grad-CAM
figure script so `ch4_gradcam_*.png` are layer2-based; keep the new
prose and Listing 5.10 as written. Minimum alternative (honest but
weak): revert listing + prose to layer3→2×2.

**Status (14 Sept): applied, with one correction to the paragraph above.**
`src/gradcam.py` now returns `layer2[-1].conv2` and every size claim in it
was rewritten to the measured values; the draft's §5.11 prose and Listing
5.10 match it verbatim. One claim here was wrong: the published
`ch4_gradcam_*.png` figures were *not* drawn from layer3. The only caller,
`plot_gradcam_grid()`, passes `arch='shallow'`, so the ResNet branch this
finding is about had never been exercised and the panels always came from
ShallowCNN's `features[8]` (4×4). The defect was real but latent — false
comments in a dead branch plus a printed listing that did not match the
repository — and the figures are therefore unchanged in content.

### M4. Rethought and downgraded → see first item under MODERATE

On the author's objection I re-examined the §5.1 StandardScaler passage
against the pipeline instead of the other way round. Verdict: the
design is sound and the passage's general problem statement is valid —
the defect is narrower than first stated (framing + a missing bridge
sentence, not self-refutation). Full reasoning and a 3-line fix below
under MODERATE; no deletion needed.

### M5. ✅ VERIFIED FIXED — fully closed (prior flag partially retracted)

The MLP sentence now reads domain-neutral (verified), and I re-checked
the one remaining *„mrežnog saobraćaja“* occurrence: it sits in the §2
BIE paragraph (*„evidentiranje mrežnog saobraćaja, uključujući CIC-DN
benchmark“*), describing other authors' application domain — legitimate
literature context, not a leftover. No action.

### M6. ✅ VERIFIED FIXED — capacities worded correctly

Uvod now reads *„dve arhitekture različitog kapaciteta (ShallowCNN
0.62M, ResNet-18 11.2M) plus kontrola inicijalizacije (ResNet-18 od
nule)“*. No action; apply the same framing if §6 repeats the old one.

### M7. ✅ VERIFIED FIXED — duplicates claim removed

p. 26 no longer claims duplicate removal (matches the code, which has
none). No action.

### M8. Structure: numbering ✅, TOC ✅; abstract still missing — exact template below

- Duplicated §5.1 → renumbered 5.2/5.3. Verified.
- TOC no longer prints field errors; dead 6.2/6.3 entries removed.
  Verified in the newest export.
- Still no **Sažetak/Abstract**. Insert both before §1 (one page
  total). Paste-ready skeleton — fill the brackets from the Appendix:

> *Sažetak.* U radu se porede četiri postupka konverzije tabelarnih
> podataka u slike (naivno pakovanje, DeepInsight, IGTD, TINTO) na tri
> skupa (Breast Cancer Wisconsin, Dry Bean, Adult Income) i tri
> konfiguracije modela (ShallowCNN, ResNet-18 pretrenirani, ResNet-18 od
> nule) — ukupno 45 eksperimenata uz stratifikovanu podelu 70/10/20
> (seed 42). Najbolja ćelija je [dry_bean/naive/shallow 93,99 % F1];
> ćelije koje nose tvrdnje ponovljene su kroz pet semena i prijavljene
> kao srednja vrednost ± sd. [Jedna rečenica glavnog nalaza po skupu +
> zaključak o transfernom učenju.] Ključne reči: [5–6].

> Mirror the same in English as *Abstract* (past tense for what was
> done, present tense for what the results show).
- Figure refs 5.1–5.5 now point at real EDA figures for this study
  (improvement vs v1); re-verify numbering after §6 figures land.

---

## MODERATE

### M4 (rethought). ✅ APPLIED AS RECOMMENDED — one grammar nit left

The author applied the fix as suggested (conditional framing +
*„Upravo zato se u ovom radu StandardScaler koristi samo kao
predobrada … statistikama sa trening skupa“*). Verified in the newest
export. Remaining micro-fix, exact:

- OLD: *„StandardScaler kao primarni mehanizma normalizacije bi uveo
  značajne matematičke nedoslednosti“*
- NEW: *„StandardScaler bi kao primarni mehanizam normalizacije uveo
  značajne matematičke nedoslednosti“*

(grammatical case + word order). Optional strengthener: append
*„(v. §5.3)“* to the bridge sentence so the prescription visibly lands
on the per-method mapping. Background reasoning for the record: the
passage's general trap (direct z-score→pixel mapping) is real, and the
pipeline's train-statistics [0,1] stage is its remedy — the design was
never unsound, only unbridged.

### m1. ✅ APPLIED — one stray quote left

Blur sentence present as recommended. Micro-fix: it ends *„…za
konvoluciona jezgra 3×3.““* — a closing `“` with no opening quote.
Exact fix: delete the stray `“` (or quote the whole inserted sentence
properly).

### m2. ✅ VERIFIED FIXED — DeepInsight fit scope now „trening skup“

The *„čitav skup podataka“* leakage-shaped phrasing is gone (p. 12).
No action.

### m3. ✅ VERIFIED FIXED — strategy table relabeled „Strategije koje literatura predlaže“

No longer reads as the study's own prescriptions. Consider the
one-sentence rationale for the tested four (TINTOlib availability, CPU
feasibility) — optional.

### m4. ✅ APPLIED — verify one possible editing casualty

Both edits present as recommended. Check-item: the sentence now reads
*„…transformacije Weight of Evidence (WoE) **l** target encoding
pristupa…“* — confirm it says ***ili*** (*„WoE) ili target“*); the
lone `l` looks like an editing casualty (possibly a PDF-extraction
artifact — verify in Word, fix if real).

### m5. ✅ VERIFIED FIXED — 128-canvas sentence removed

The unevidenced *„povećanje platna na 128 ne pomaže“* claim is gone
from the newest export (confirmed by search). No action; do not
reintroduce without running the diagnostic.

### m6. F1-comparability caveat (C6) still undisclosed — exact paragraph + action

Re-verified: local CNN JSONs still show `f1_macro_all: None`. Two steps:

1. Code action first: run `backfill_metrics.py --write` locally, or the
   comparable column stays empty for all 36 CNN cells in every §6 table.
2. Insert after *„Primarna metrika za rangiranje je F1.“* (p. 37):
   - NEW: *„Zabeleženi ključ `f1_macro` je makro-F1 samo na Dry Bean
     skupu; na binarnim skupovima to je F1 pozitivne klase, pa se
     poređenja između skupova iskazuju makro-F1 prosekom preko svih
     klasa (`f1_macro_all`) i balansiranom tačnošću, dopunjenim iz
     matrica konfuzije.“*

### m7. MLP-validation wording (p. 27) — exact replacement

- OLD: *„…isključivo na trening redu (istim redovima kao CNN,
  validacija se koristi samo za rano zaustavljanje CNN modela i nikada
  nije deo treninga baselajna).“*
- NEW: *„…isključivo na redovima trening skupa kao i CNN (RF i XGBoost
  na celom trening skupu, dok MLP unutar njega izdvaja internih
  stratifikovanih 10% za rano zaustavljanje; validacioni skup CNN modela
  nikada nije deo treninga baselajna).“*

(matches `src/baselines/mlp.py:27-33`: `early_stopping=True,
validation_fraction=0.1` on `X_train`).

### m8. Bound explicitly permits the broken case — exact action

`requirements.txt` pins **`scikit-learn>=1.3.0`**, i.e. the fallback
branch (`mlp.py`: unweighted training on < 1.4) is within the declared
environment. Change the bound to **`scikit-learn>=1.4`**; then grep the
Colab logs for `[mlp] … without per-sample balancing` — if that line
appears, the MLP baseline (weakest on Adult, 68.51) ran handicapped and
must be rerun. No thesis-text change needed once pinned.

### m9. ViT branch in Listing 5.7 — exact trim

Delete these three lines from the listing:

```
    elif arch == 'vit':
        from src.models.vit_wrapper import ViTWrapper
        return ViTWrapper(num_classes=num_classes, pretrained=True, input_channels=3)
```

and add beneath the listing: *„ViT arhitektura je implementirana
(`src/models/vit_wrapper.py`), ali je izostavljena iz matrice jer sa
~830 s/epohi na CPU nije izvodljiva bez GPU vremena.“* (per
`run_all.py:41-46`).

### m10. „TINTOlib 1.3.1“ unpinned — exact action

`requirements.txt:37` lists bare `TINTOlib`, yet layouts are
version-sensitive (the file's own header says drift *„changes the T2I
layouts“*). Run `pip freeze | grep -Ei 'tinto|torch|scikit-learn|
xgboost|numpy|pandas'` in the Colab session that produced the grid,
then pin the exact strings (`TINTOlib==…`, …). If the frozen version is
not 1.3.1, correct Tables 5.2/§5 to the version actually used.

### m11. ✅ APPLIED — one doubled period left

§7 now reads *„poput bikubične interpolacije“* (verified). Micro-fix:
the sentence ends *„…interpolacije).. Pored…“* — delete the extra `.`
(updated text kept the old parenthesis-dot and added another).

### m12. §7 concludes nothing from this thesis — exact skeleton

Keep the literature-review paragraphs, then replace the generic closing
with four paragraphs following this skeleton (fill brackets from the
Appendix):

1. *Šta je urađeno:* 45 eksperimenata (4 T2I × 3 skupa × 3 konfiguracije
   + 9 baselajna), podela 70/10/20 seed 42, ponavljanje tvrdnji kroz 5
   semena.
2. *Glavni brojevi:* najbolja ćelija [dry_bean/naive/shallow 93,99 %];
   na Adult/naive nema izmerenog transfera [Δ −0,28 pp, interval sadrži
   nulu]; jedina razlučiva razlika tvrdnji je naive nad TINTO na
   Adult/ShallowCNN [+2,15 pp].
3. *Ograničenja:* jedan run po ćeliji glavne tabele; odbačeni artefakt
   checkpointa (epoha 2/50) kao opomena; C11 združenost prijavljena
   eksplicitno.
4. *Pravci:* [iz postojećeg teksta, bez LCG/saliency kao da su korišćeni
   — *„kao budući pravac“*].

Exact micro-fix: *( HDLS)* → *(HDLSS)* (p. 39).

---

## MINOR / structural notes (references out of scope, listed only where broken)

- Text cites **[50]** (§4.1) but the bibliography ends at **[43]** —
  renumber or complete entries when §6 lands.
- §8.1 *Prilozi* is gone from the body (references now end at [43])
  but the **TOC still lists 8.1 PRILOZI → p. 42** — dangling entry.
  Exact fix: either delete the TOC line (update the field), or — better
  — add a one-page appendix with the study's data/code links (UCI URLs,
  repo + commit hash of the grid) and keep the entry.
- Adult dual-N clarity (new minor): the overview table says 48.842
  samples / 14 features while Table 5.1 says 45.222 / 104, with the
  connection (*−3.620 „?“ rows*, code comment verified) left implicit.
  Exact fix — extend the Table 5.1 Adult note to: *„8 kategoričkih + 6
  numeričkih; 75:25; od zvaničnih 48.842 redova nakon uklanjanja 3.620
  redova sa „?“ ostaje 45.222“*.
- Proofreading debris for the §6 rewrite pass: *„pomoću“* (dangling,
  p. 37), *„glavne tabele“* (typo), *„distibucije“* (p. 23),
  *„distribution shifits“* (p. 7), doubled period after the leakage
  bullet (p. 21).
- Verified-clean, keep as is: 45-experiment matrix; split procedure +
  Listing 5.1; Adult 45222/104/8+6/`?`-removal/UCI-merge; class ratios
  63:37, 75:25 (majority 75.2%), dry-bean 6.8:1 (recomputed from test
  confusion matrices: 709:104); baseline configs incl.
  `random_state=42` everywhere; CNN hyperparameters + Listings
  5.2/5.8; T2I parameter table (naive grid/bicubic, IGTD
  Pearson+Euclid/squared/1000/50, TINTO PCA/blur params, TINTOlib
  seeds); seeds 42–46 design; thresholds 0.02/0.01
  (`src/ablation.py:212,512`); ShallowCNN 618,178 params and valid
  `features[8]`; T4 plausibility from `train_time_sec`.

---

## Appendix — corrected key numbers for the §6 rewrite

⚠️ Supersedes the first review's appendix: commit `802bb17` discarded
the −11.40 pp headline as a checkpoint artifact (adult/naive/resnet's
stored run kept an epoch-2 checkpoint; the cell repeats at **68.36 ±
0.82 %**). Do **not** report 57.58 % / −11.40 pp as findings.

- Best cell overall: **dry_bean / naive / shallow 93.99** (stored F1).
- Adult / naive: pretrained **68.36 ± 0.82 %** (acc 80.97 ± 1.62) vs
  scratch **68.64 ± 0.75 %** (main-table single run 68.98) → paired
  Δ **−0.28 pp [−1.55; +0.98]**, indistinguishable from noise; macro-F1
  Δ +0.14 pp [−1.22; +1.49]; pretrained bal-acc 81.59 ± 0.64 % (above
  the 75.2 % majority — the old run's below-majority accuracy was the
  artifact's fingerprint). Transfer learning on T2I: **absence of
  effect**, not harm; pretraining effect within ±1 pp on all sets.
- Adult / shallow: naive 68.94 vs tinto 66.90 vs deepinsight 66.34
  (collision cost, ~2–3 pp) vs igtd 68.52; baselines mlp 68.51, rf
  69.99, xgb 71.43. Paired across seeds (commit `7b763df`):
  **naive-over-TINTO +2.15 pp [+1.54; +2.76], resolved** — the one
  claim-bearing difference that survives the noise interval; report it
  as the collision-cost finding.
- Breast / tinto / shallow 96.45 — compare against baselines in
  `f1_macro_all` only (see m6).
- Spreads: `results/seed_summary.csv` (Colab run); stability table
  procedure: `scripts/stability_table.py` → `results/stability_table.md`
  (rad-nacrt §6.7). The §6 transfer paragraph template (with the
  combined-effect disclosure and the §4.7-control pointer) is drafted
  in `Plan/seminar2-rad-nacrt.md` §6.1.3/§6.3 — port it, don't
  re-derive it.
