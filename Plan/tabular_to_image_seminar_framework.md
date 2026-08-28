# Experimental Framework: Tabular-to-Image Conversion using CNNs

To match the impressive scope of your first seminar while shifting the focus toward deep implementation details, you need a structured grid of experiments. Comparing architectures, transformation techniques, and data complexities will give you the "girth" your professor expects.

Here is a comprehensive framework for your practical implementation.

### 1. The Experimental Design Matrix
Instead of just testing datasets, you will vary **three core dimensions**: Tabular-to-Image (T2I) methods, CNN architectures, and dataset types. 

```
                                [ Your Seminar Scope ]
                                          │
         ┌────────────────────────────────┼────────────────────────────────┐
         ▼                                ▼                                ▼
  [ 3 T2I Methods ]              [ 3 CNN Architectures ]         [ 3 Dataset Types ]
  • IGG (Image Grid)             • Custom Shallow CNN            • High-cardinality categorical
  • DeepInsight                  • ResNet (Transfer Learning)    • Highly correlated numerical
  • t-SNE / UMAP Embedding       • Vision Transformer (ViT)      • Imbalanced class data
```

### 2. What to Implement (The Core Pipeline)
Your implementation chapter should document the heavy engineering required to make this pipeline work. Focus on these three areas:

#### A. Tabular-to-Image Transformation Strategies (Implement 3)
*   **Naive Pixel Arrangement (Baseline):** Reshape features directly into a 2D grid (padded with zeros if necessary). This acts as your control group to prove if advanced spatial ordering actually matters.
*   **DeepInsight Pipeline:** Implement the classic DeepInsight methodology. Use t-SNE or PCA to find feature similarities in 2D space, create a bounding box, and map feature intensities to pixel clusters.
*   **Distance-Based Embedding (UMAP):** Use UMAP to project feature distances into 2D pixel coordinates, handling non-linear relationships better than PCA.

#### B. Architectural Variations (Implement 3)
*   **Custom Shallow CNN:** Built from scratch to show you understand kernel sizes, stride, and pooling for abstract feature maps.
*   **ResNet-18 (Transfer Learning):** Initialize with ImageNet weights. Discuss the implementation challenge of adapting 1-channel grayscale T2I images to 3-channel RGB networks.
*   **Vision Transformer (ViT) or ConvNeXt:** A modern architecture to evaluate if patch-based attention mechanisms handle transformed tabular data better than standard convolutions.

#### C. Dataset Complexity Variations (Use 3)
*   **Dataset 1 (High-Cardinality Categorical):** Tests how embedding layers or one-hot encoding affect image spatial structures.
*   **Dataset 2 (Multiclass / Highly Correlated):** Tests if the CNN accurately captures geometric patterns created by highly correlated features.
*   **Dataset 3 (Extreme Class Imbalance):** Focuses your implementation on advanced loss functions (Focal Loss) and data augmentation in the image domain versus the tabular domain.

---

### 3. Deep Implementation Metrics to Compare
Since this seminar focuses more on implementation details than the last one, expand your evaluation metrics beyond basic accuracy or F1-scores. 

| Metric Category | Specific Metrics to Compare | Why it Adds "Girth" |
| :--- | :--- | :--- |
| **Computational Footprint** | Training time per epoch, GPU VRAM utilization, total parameters. | Proves the engineering viability of converting data to images. |
| **Data Efficiency** | Performance at 10%, 50%, and 100% training data size. | Analyzes if CNNs require significantly more data than traditional ML. |
| **Robustness** | Performance drop when adding Gaussian noise to pixels vs. tabular features. | Evaluates model stability and structural integrity of the transformation. |
| **Explainability (XAI)** | Grad-CAM activation maps superimposed over the generated feature images. | Visually proves whether the CNN is learning real feature interactions or just noise. |

---

### 4. Crucial Implementation Validation: The Ablation Study
To make this a true academic seminar, you must validate that the CNN is leveraging spatial patterns. Implement a **Pixel Shuffling Test**:
1. Train your best T2I + CNN pipeline normally.
2. For the test set, **randomly permute (shuffle) the pixel positions** using a fixed key across all images.
3. If the CNN accuracy drops significantly, it proves the transformation successfully encoded meaningful spatial structures. If accuracy stays the same, the network is merely treating the image as a flattened vector, defeating the purpose of the CNN.

---

### 5. Structuring the Seminar Paper
Organize your paper to highlight the engineering effort:
1.  **Introduction & Paradigm Shift:** Why convert tabular data to images? (e.g., capturing non-linear interactions spatializing relationships).
2.  **The Pipeline Architecture:** Detailed math and algorithms behind DeepInsight and UMAP coordinate mapping.
3.  **Implementation Details:** Hardware used, deep learning framework (PyTorch/TensorFlow), optimization bottlenecks, and hyperparameter tuning strategy.
4.  **Experimental Results:** Cross-comparison tables matching the 3 transformation methods against the 3 CNN models across all 3 datasets.
5.  **Ablation Study:** What happens if you shuffle the pixel positions after transformation? (This proves if the spatial structure actually matters to the CNN).
