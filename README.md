# 🧬 EGFR Bioactivity Prediction using Cheminformatics & Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project explores the prediction of **bioactivity** (IC₅₀) for compounds targeting the **Epidermal Growth Factor Receptor (EGFR)** using molecular fingerprints and physicochemical descriptors. The goal is to distinguish between **active and inactive compounds** with respect to their inhibitory effect on EGFR, a critical target in cancer therapy.

> 🚀 Built using **Python**, **RDKit**, **scikit-learn**, and **Optuna**, this project is a reproducible pipeline from raw SMILES strings to an optimized Random Forest classifier.

---

## 📚 Table of Contents

- [📍 Motivation](#-motivation)
- [🧪 Dataset](#-dataset)
- [🛠️ Methodology](#️-methodology)
- [📊 Results](#-results)
- [🧠 Key Insights](#-key-insights)
- [🧬 Biological Relevance](#-biological-relevance)
- [🔮 Future Work](#-future-work)
- [⚙️ Installation](#️-installation)
- [📁 Repository Structure](#-repository-structure)
- [🤝 Feedback Welcome](#-feedback-welcome)

---

## 📍 Motivation

Modern drug discovery is **costly**, **time-consuming**, and faces a high attrition rate in clinical pipelines. **Computer-Aided Drug Design (CADD)** can help streamline this process by leveraging cheminformatics and machine learning to:

- Prioritize candidate molecules before synthesis.
- Reduce the cost of wet-lab screening.
- Offer interpretable insights into structure-activity relationships (SAR).

This project aims to **perform a practical application of CADD** for bioactivity prediction, starting from raw chemical data and ending with a robust machine learning model.

---

## 🧪 Dataset

- **Source**: [ChEMBL Database](https://www.ebi.ac.uk/chembl/)
- **Target Protein**: EGFR (Epidermal Growth Factor Receptor)
- **Activity Metric**: IC₅₀ values
- **Size**: ~1000 curated compounds after preprocessing

Bioactivity was binarized:
- `Active (1)` if pIC₅₀ ≥ 6
- `Inactive (0)` otherwise

---

## 🛠️ Methodology

The pipeline includes:

1. **Data Curation**:
   - Filtering for standard, human-targeted assays
   - Handling duplicates, removing unreliable data points

2. **Feature Engineering**:
   - Molecular fingerprints (1024-bit Morgan Fingerprints via RDKit)
   - Physicochemical descriptors: MW, LogP, NumHDonors, NumHAcceptors

3. **Modeling**:
   - **Random Forest Classifier** with `class_weight="balanced"`
   - Hyperparameter tuning using **Optuna**
   - Validation via **Nested Cross-Validation** and hold-out test set

4. **Evaluation**:
   - F1-score, ROC-AUC, learning curve, confusion matrix, and feature importances

---

## 📊 Results

**Final Evaluation on Test Set**:
| Metric           | Value   |
|------------------|---------|
| Accuracy         | 91%     |
| F1-Score         | 0.94    |
| ROC-AUC          | 0.89    |
| Minority Class F1 (Inactive) | 0.75 |

**Hyperparameters Used**:
```json
{
  "n_estimators": 105,
  "max_depth": 12,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "log2"
}
```

---

## 🧠 Key Insights

- **Fingerprint features dominate** in feature importance — confirming that specific substructures are vital in EGFR inhibition.
- **Physicochemical descriptors**, particularly LogP and MW, still ranked high (8th and 11th), reinforcing traditional SAR relevance.
- **class_weight='balanced'** was crucial to achieve fairness across classes. Without it, the model overestimated performance on active compounds.

---

## 🧬 Biological Relevance

- **EGFR** is a validated cancer target involved in cell proliferation and survival.
- This model mimics early-stage **virtual screening**: prioritizing which molecules should be tested in vitro.
- False negatives (actives missed) could be **novel scaffolds**, while false positives might indicate **structurally similar but biologically inactive compounds** — important leads for further research.

---

## 🔮 Potential Future Work

- Include **3D structure-based descriptors** or docking scores for richer feature sets.
- Try **multitask models** for related kinase families.
- Use **confidence-based predictions** (e.g., conformal prediction).
- Explore **ensemble stacking** or **graph neural networks** for deeper learning.

---

## ⚙️ Installation

Install required packages:

```bash
pip install -r requirements.txt
```

*Optionally*: The method to install **Conda** is given in the second notebook.

---

## 📁 Repository Structure

```
.
├── data/
│   └── raw_and_cleaned_data.csv  # kept in repo for reference
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_feature_engineering.ipynb
│   └── 3_model_training_evaluation.ipynb
├── figures/
│   └── *.png  # plots and visualizations used in the notebook
├── .gitignore
├── Requirements.txt
└── README.md
```

---

## 🤝 Feedback Welcome

This project is a learning experience and a stepping stone in cheminformatics and CADD.  
**Suggestions, critiques, and improvements are warmly welcome!**

---

## 🙏 Thank You

Thanks for checking out the project! 
