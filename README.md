```markdown
# DeepEnMap

**Multi-Modal Deep Learning for Ordinal Energy Poverty Risk Mapping**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeepEnMap predicts energy poverty risk by fusing satellite imagery with demographic data. Instead of treating risk levels as unordered categories, it treats them as what they really are — an ordinal scale — using a custom **Distance-Penalized Cross-Entropy (DPCE)** loss that penalizes far-off wrong predictions more than close ones, evaluated with a companion **Mean Absolute Ordinal Error (MAOE)** metric.

![Model Architecture](DeepEnMap%20Latest%20Materials/Images/03_model_architecture.png)
*Figure 1 — DeepEnMap architecture: satellite imagery is processed by a CNN branch, demographic features by a dense branch; both are fused and classified using the DPCE loss.*

---

## Table of Contents

- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
- [Dataset Overview](#dataset-overview)
- [Results](#results)
- [Ablation Studies](#ablation-studies)
- [Computational Cost](#computational-cost)
- [Explainability](#explainability)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Citation](#citation)

---

## Key Contributions

- **Distance-Penalized Cross-Entropy (DPCE) Loss** — an ordinal-aware loss that penalizes probability mass placed on ordinally distant wrong classes more heavily than on nearby ones, while remaining a strict, backward-compatible generalization of standard cross-entropy (λ = 0 recovers CE exactly).
- **Mean Absolute Ordinal Error (MAOE)** — an evaluation metric that captures *how far* predictions deviate from the truth (in risk-level steps), not just whether they're wrong.
- **Multi-modal fusion architecture** — a 3-block CNN over satellite patches fused with a dense demographic encoder, jointly trained end-to-end.
- **Grad-CAM explainability** — saliency maps computed with respect to the DPCE objective, so interpretability reflects the ordinal-aware training signal.

## Methodology

| Algorithm | Name | Function |
|---|---|---|
| 1 | Data Preprocessing | Normalizes satellite patches, standardizes demographic features, one-hot encodes labels, 80/20 train-val split |
| 2 | Spatial Feature Extraction (CNN) | 3-block CNN (32→64→128 filters) → 256-D feature vector |
| 3 | Multi-Modal Fusion & Classification | Concatenates 256-D image features + 64-D demographic features → 7-class softmax |
| 4 | End-to-End Training | Adam optimizer, DPCE loss, early stopping (patience = 3) |

**DPCE Loss:**

$$\mathcal{L}_{\text{DPCE}} = -\frac{1}{B}\sum_{i=1}^{B} \log(\hat{Y}[i,y_i]+\epsilon) \;+\; \frac{\lambda}{B}\sum_{i=1}^{B}\sum_{c=1}^{K} \frac{|y_i-c|}{K-1}\hat{Y}[i,c]$$

**MAOE Metric:**

$$\text{MAOE} = \frac{1}{N}\sum_{i=1}^{N} |\hat{c}_i - y_i|$$

## Dataset Overview

![Class Distribution](DeepEnMap%20Latest%20Materials/Images/01_class_distribution.png)
![Sample Grid](DeepEnMap%20Latest%20Materials/Images/02_sample_grid.png)

*Left: distribution of samples across the 7 ordinal risk classes. Right: representative satellite patches per class.*

## Results

Trained for up to 10 epochs (early stopping, patience = 3) across 5 random seeds. Reported as mean ± std.

| Loss (λ) | Accuracy | MAOE ↓ | F1 (macro) |
|---|---|---|---|
| Standard CE (λ = 0) | 0.9157 ± 0.0099 | 0.1229 ± 0.0151 | 0.9096 ± 0.0105 |
| **DPCE (λ = 1)** | **0.9166 ± 0.0092** | **0.1201 ± 0.0130** | **0.9102 ± 0.0104** |

DPCE achieves comparable accuracy to standard cross-entropy while reducing MAOE — i.e., when the model is wrong, it tends to be wrong by a *smaller* ordinal margin.

![Training History](DeepEnMap%20Latest%20Materials/Images/04_training_history.png)
![Confusion Matrix](DeepEnMap%20Latest%20Materials/Images/05_confusion_matrix.png)

*Training/validation curves (left) and confusion matrix (right) show most confusions fall on adjacent risk classes, consistent with the ordinal structure the DPCE loss is designed to exploit.*

**Error severity distribution (CE vs. DPCE):**

| Ordinal Distance | CE (%) | DPCE (%) | Relative Change |
|---|---|---|---|
| 0 (Correct) | 92.80 | 92.57 | −0.24% |
| 1–2 (Close) | 6.93 | 7.02 | +1.34% |
| 3–4 (Moderate) | 0.28 | 0.41 | +46.67%* |
| 5–6 (Far) | 0.00 | 0.00 | — |

<sub>*Off a very small base rate (≈0.28%); absolute increase is 0.13 percentage points.</sub>

![Ordinal Error Distribution](DeepEnMap%20Latest%20Materials/Experiment_4/exp4_ordinal_error_distribution.png)
![Misclassified Examples](DeepEnMap%20Latest%20Materials/Images/06_misclassified_examples.png)

## Ablation Studies

### Modality Ablation

| Modality | Accuracy | MAOE ↓ | F1 (macro) | Parameters |
|---|---|---|---|---|
| Demographic only | 0.5827 ± 0.0061 | 0.4829 ± 0.0083 | 0.4780 ± 0.0096 | 9,351 |
| Image only | 0.8593 ± 0.0033 | 0.3280 ± 0.0146 | 0.8503 ± 0.0029 | 2,193,351 |
| **Multi-modal (fused)** | **0.9166 ± 0.0092** | **0.1201 ± 0.0130** | **0.9102 ± 0.0104** | 2,202,695 |

Fusing both modalities substantially outperforms either alone — demographic data adds meaningful signal despite contributing <0.5% of total parameters.

### λ Sensitivity Sweep

![Lambda Sensitivity](DeepEnMap%20Latest%20Materials/Experiment_2/exp2_lambda_sensitivity.png)

Statistical significance was assessed via paired t-tests with Holm–Bonferroni correction across λ ∈ {0.5, 1.0, 2.0, 5.0} against the λ = 0 baseline (5 seeds each). At λ = 5.0, the accuracy improvement (p ≈ 0.028, Cohen's d ≈ 1.50) and MAOE reduction (p ≈ 0.030, Cohen's d ≈ −1.47) reach significance under this correction; other λ values did not reach significance at this sample size.

## Computational Cost

| Modality | Parameters | FLOPs / Inference |
|---|---|---|
| Demographic only | 9,351 | 18,538 |
| Image only | 2,193,351 | 87,692,394 |
| Multi-modal (fused) | 2,202,695 | 87,710,890 |

The demographic branch adds <0.1% computational overhead relative to the image branch, making multi-modal fusion essentially "free" in inference cost while delivering the largest accuracy and MAOE gains.

## Explainability

Grad-CAM saliency maps are computed with respect to the DPCE loss (Eq. above), so activation regions reflect the ordinal-aware training objective rather than a distance-agnostic one.

![Grad-CAM Heatmaps](DeepEnMap%20Latest%20Materials/Images/07_gradcam_heatmaps.png)
![Grad-CAM Samples](DeepEnMap%20Latest%20Materials/Images/gradcam_samples.png)

**Spatial Risk Predictions**

![Country Prediction Map](DeepEnMap%20Latest%20Materials/Images/08_country_prediction_map.png)

## Repository Structure

```
DeepEnMap Latest Materials/
├── Codes and files/
│   ├── DeepEnMap_Experiments.ipynb       # Full experiment notebook
│   ├── DeepEnMap_Experiments.ipynb - Colab.pdf
│   └── deepenmap_experiments.py          # Script version
├── Experiment_1/                          # Baseline (CE) vs DPCE comparison
├── Experiment_2/                          # λ sensitivity sweep + significance tests
├── Experiment_3/                          # Modality ablation
├── Experiment_4/                          # Ordinal error distribution analysis
├── Images/                                 # Figures used in this README
└── computational_cost/                    # Parameter / FLOP comparisons
```

## Requirements

- Python 3.x
- TensorFlow 2.15
- Keras

All experiments were run on a single NVIDIA T4 GPU (16 GB VRAM) via Google Colaboratory.

## Citation

```bibtex
@article{deepenmap2026,
  title={DeepEnMap: A Multi-Modal Deep Learning Framework for Energy Poverty Risk Mapping with Ordinal-Aware Loss},
  author={Ahmed, Sarder Junaid},
  year={2026}
}
```

---

## 👨‍💻 About the Developer

<div align="center">

<img src="https://avatars.githubusercontent.com/Junaid-Ahmed-Rupok" width="100" style="border-radius:50%"/>

### Sarder Junaid Ahmed
**Data Scientist & Machine Learning Engineer**

*Transforming complex data into strategic decisions through rigorous statistical modeling and production-ready machine learning systems.*

[![GitHub](https://img.shields.io/badge/GitHub-Junaid--Ahmed--Rupok-181717?logo=github)](https://github.com/Junaid-Ahmed-Rupok)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sarder%20Junaid%20Ahmed-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sarder-junaid-ahmed-059b68240/)
[![Portfolio](https://img.shields.io/badge/Portfolio-junaid--ahmed--rupok.github.io-1E88E5?logo=githubpages&logoColor=white)](https://junaid-ahmed-rupok.github.io/__portfolio__Yes/)
[![Email](https://img.shields.io/badge/Email-junaidahmedrupok%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:junaidahmedrupok@gmail.com)

</div>

**Specializations:** Statistical ML · Causal Inference · Trustworthy AI · Fairness-Aware ML · RAG Systems

**Selected Research:**
- 📄 **Ahmed, S.J.** et al. (2026). *Machine Learning for Crime Classification: A Fairness-Aware Approach to Class Imbalance.* Journal of Machine Learning and Applications, 2(1), 9–17. [DOI: 10.61577/jmla.2026.100002](https://doi.org/10.61577/jmla.2026.100002)
- 📄 **Ahmed, S.J.** et al. (2026). *CF-EGAT: A Causal Fairness-Aware Equity Graph Attention Network for Country-Level Environmental Livability Classification.* SPECTRA 2026. 🏆 **1st Best Paper Award**
- 📄 **Ahmed, S.J.** (2025). *Multi-Dimensional Statistical Similarity for Governance Classification: Beyond Arbitrary Thresholds.* APMEE 2025. 🏆 **Best Research Paper Award**

**Other Deployed Projects:**
- 🔬 [ReproHub](https://reproapp-8jb7vbhnqyltxq23bsr8xn.streamlit.app/) — Automated research reproducibility platform with composite scoring across 11 statistical tests
- 📊 [StatsPro](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app/) — AI-powered statistical analysis platform with automated CSV-to-report workflows

**Honors:**
🏆 1st Best Paper — SPECTRA 2026 &nbsp;·&nbsp;
🏆 Best Research Paper — APMEE 2025 &nbsp;·&nbsp;
🎖️ Esteemed Alumni Award — YLRL RUET 2024 &nbsp;·&nbsp;
⭐ Perfect GPA 5.00/5.00 — SSC & HSC &nbsp;·&nbsp;
🎓 National Merit Scholarship — 2009 & 2013

---

## 📄 License

MIT — see [LICENSE](LICENSE).
```
