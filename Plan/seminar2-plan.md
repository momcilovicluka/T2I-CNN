# Seminar 2: Tabular-to-Image Conversion with CNNs — Final Plan

## Professor Requirements (from email chain)

- **Deep learning continuation** of ML seminar (IDS with 10 supervised models)
- **Tabular → Image → CNN** — topic feels fresh, no one has covered it before
- **Transfer learning vs. locally trained** — Luka's suggestion, professor approved
- **"More tangible work"** — heavy implementation, not just theory
- **Similar scope to ML seminar** — comprehensive but manageable
- **Generic benchmark datasets** — no specific domain required

---

## Source Plans Evaluated

| Aspect | GPT Plan | Gemini Plan | **Final Decision** |
|--------|----------|-------------|-------------------|
| Datasets | 4 | 3 | **3** (manageable, sufficient variety) |
| T2I Methods | 4 (custom implementations) | 3 | **3** (use TINTOlib library, not from scratch) |
| CNN Architectures | 5 custom CNNs | 3 (incl. ResNet, ViT) | **3** (shallow + ResNet-18 + ViT) |
| Transfer Learning | ❌ Missing | ✅ ResNet + ViT | **✅ Keep** (professor's key request) |
| Tabular Baselines | ✅ 4 models | ❌ Missing | **✅ Keep** (3 models) |
| Ablation Study | ✅ Feature ordering | ✅ Pixel shuffling | **✅ Keep** (both merged) |
| Grad-CAM | ❌ | ✅ | **Nice-to-have** (add if time permits) |
| Total Experiments | 80+ | 27 | **36** (27 CNN + 9 baselines) |

---

## Final Scope: 3 × 3 × 3 + Baselines + Ablation

### Datasets (3 — chosen for variety)

| # | Dataset | Type | Features | Why |
|---|---------|------|----------|-----|
| 1 | **Breast Cancer Wisconsin** (sklearn) | Binary, 30 numerical | Small (569 samples) | Clean baseline, fast experiments |
| 2 | **Dry Bean** (UCI) | Multiclass (7), 16 numerical | Large (~13K samples) | Tests scalability, multiclass handling |
| 3 | **Adult Income** (UCI) | Binary, 14 mixed (categorical + numerical) | Medium (~48K samples) | Tests categorical encoding impact on images |

### Tabular-to-Image Methods (3)

| # | Method | Implementation | Rationale |
|---|--------|---------------|-----------|
| 1 | **Naive Reshape** (baseline) | Custom — pad/reshape feature vector to square grid | Simplest possible; proves if spatial ordering matters |
| 2 | **DeepInsight** | `TINTOlib` library (`pip install tintolib`) | Well-cited method; validates pipeline against published results |
| 3 | **IGTD** | `TINTOlib` or implement rank-based permutation | Different algorithmic family (distance-preserving vs. manifold) |

### CNN Architectures (3)

| # | Architecture | Type | Purpose |
|---|-------------|------|---------|
| 1 | **Shallow CNN** (2-3 conv layers) | Locally trained | Shows understanding of CNN basics from scratch |
| 2 | **ResNet-18** (pretrained ImageNet) | Transfer learning | Key comparison point (professor's request) |
| 3 | **ViT-base** (pretrained) | Transfer learning (attention-based) | Modern architecture; tests if attention > convolutions |

### Tabular ML Baselines (3)

| # | Model | Purpose |
|---|-------|---------|
| 1 | **Random Forest** | Standard ensemble baseline |
| 2 | **XGBoost** | State-of-the-art tabular baseline |
| 3 | **MLP** (simple feedforward NN) | Neural network baseline without spatial structure |

**Total: 3×3×3 = 27 CNN experiments + 3×3 = 9 tabular baselines = 36 experiments**

---

## Implementation Pipeline

```
Dataset → Preprocessing → [Naive | DeepInsight | IGTD] → Image Generation
                                                                    ↓
                                              [Shallow CNN | ResNet-18 | ViT] → Training
                                                                    ↓
                                              Evaluation (8 metrics) → Comparison Tables
```

Separate tabular baseline experiments:
```
Dataset → Preprocessing → [RF | XGBoost | MLP] → Evaluation
```

---

## Metrics

**Classification:**
- Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC

**Computational:**
- Training time, Number of parameters, Inference time

**Explainability (if time permits):**
- Grad-CAM heatmaps on ResNet-18 (visual proof the CNN uses spatial patterns)

---

## Ablation Study

Take best-performing T2I method + ResNet-18, then:

1. **Full features** → baseline accuracy
2. **Remove highly correlated features** → does performance drop?
3. **Random feature permutation** → does destroying spatial structure hurt CNN?
4. **Preserve correlated features spatially** → does explicit clustering help?

Core academic finding: *Whether spatial organization of features actually matters for CNN performance.*

---

## Paper Structure

### 1. Introduction
- Why convert tabular data to images? What CNNs gain from spatial structure.
- Research question: *How does the T2I transformation method affect CNN classification performance?*

### 2. Background & Related Work
- Survey of T2I methods (DeepInsight, IGTD, REFINED, SuperTML, etc.)
- CNN architectures for non-natural images
- Existing benchmarks and libraries (TINTOlib, etc.)

### 3. Methodology
- Pipeline architecture (diagram)
- T2I algorithms (math for DeepInsight t-SNE mapping, IGTD rank permutation)
- CNN architectures (shallow CNN design, ResNet-18 adaptation, ViT adaptation)
- Preprocessing (normalization, categorical encoding, train/val/test splits)

### 4. Experimental Setup
- Datasets description and statistics
- Hardware/software environment (GPU, PyTorch version, etc.)
- Hyperparameter strategy (grid search, early stopping, fixed seeds)
- Reproducibility details

### 5. Results
- **Table 1:** All 27 CNN experiment results (3 datasets × 3 T2I × 3 CNNs)
- **Table 2:** Tabular baselines vs. best CNN results
- **Table 3:** Computational comparison (time, parameters)
- **Figures:** Grad-CAM visualizations, confusion matrices

### 6. Analysis
- Which T2I method works best and when?
- Transfer learning vs. locally trained — when does each win?
- Does converting tabular → image beat treating it as tabular? (vs. RF/XGBoost)

### 7. Ablation Study
- Feature correlation impact
- Pixel shuffling test (does spatial structure matter?)
- Feature selection impact

### 8. Conclusion
- Key findings
- Limitations
- Future work

---

## What NOT to Do

- Don't build 4-5 custom CNN architectures — 3 is enough
- Don't implement T2I from scratch — use TINTOlib for DeepInsight/IGTD
- Don't use 4+ datasets — 3 is plenty for a seminar
- Don't include class imbalance handling (Focal Loss, GAN) — scope creep
- Don't add more than 3 tabular baselines — diminishing returns

---

## Key Differences from ML Seminar

| ML Seminar | Seminar 2 |
|-----------|-----------|
| Many models × datasets | Fewer models but deeper analysis |
| Tabular data only | Tabular → Image → CNN pipeline |
| Standard evaluation | + Explainability (Grad-CAM) + Ablation |
| No transfer learning | Transfer learning is the centerpiece |
| Implementation: train & evaluate | Implementation: build entire T2I pipeline |

---

## Estimated Timeline

| Phase | Work | Time |
|-------|------|------|
| 1 | Literature review + pipeline design | 1 week |
| 2 | Implement T2I methods + preprocessing | 1-2 weeks |
| 3 | Implement CNNs + training loop | 1 week |
| 4 | Run all 36 experiments + collect results | 1 week |
| 5 | Ablation study | 3-4 days |
| 6 | Grad-CAM + analysis + paper writing | 2 weeks |
| **Total** | | **~5-6 weeks** |

---

## Reference Files in This Workspace

- `Notebook/Tabular-to-Image Transformation Method Metadata - Table 1.csv` — method metadata from literature
- `Notebook/zotero-import.bib` — BibTeX entries for Zotero import
- `Notebook/references.bib` — full bibliography from Inciteful
- `Plan/tabular_to_image_cnn_seminar_plan(1).md` — GPT plan (original)
- `Plan/tabular_to_image_seminar_framework.md` — Gemini plan (original)
