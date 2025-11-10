# 🧬 Attribution Pipeline — Execution Guide

This repository computes atomic-level attribution scores for Drug–Target Interaction (DTI) models using multiple explainability methods (Integrated Gradients, Input × Gradient, Guided Backpropagation, and Gradient SHAP).

---

## ⚙️ Setup

To set up the environment, make sure Conda or Miniforge is installed. Then create and activate the environment from the provided file:
1. **Activate the environment**
   ```bash
   conda env create -f environment.yml
   conda activate rindti

2. **To execute the attribution pipeline:**
   ```bash
snakemake -s attribution_pipeline/Snakefile --cores 1