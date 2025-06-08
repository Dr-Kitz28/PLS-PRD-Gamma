# GPU Demand Forecasting via GitHub Parallel Code Analysis

## Project Overview

This project aims to forecast GPU demand and predict quarterly EPS (Earnings Per Share) of leading GPU manufacturers by analyzing parallelism trends in top open-source repositories on GitHub.  
We infer demand for specific GPU classes by quantifying the ratio of parallel to sequential code, map it to GPU types, and model its effect on supply-demand dynamics and financial performance using machine learning.

---

## Objectives

- Scrape and rank high-rated GPU computing repositories on GitHub.
- Measure the percentage of parallel vs sequential code across projects.
- Infer GPU model-level demand based on code structure.
- Predict quarterly EPS using a supply-demand forecasting model.

---

## Literature Basis

- **Munaiah et al. (2017)** – Proposed Reaper, a framework to automatically classify GitHub repositories as engineered software projects. Their work validates the importance of repository curation before data mining and reinforces the idea that popularity alone (e.g., stars/forks) may not reflect software engineering relevance.

- **Kamei et al. (2013)** – Conducted a large-scale study on Just-in-Time defect prediction, highlighting the predictive power of commit-level features—relevant to identifying GPU-intensive development trends in repositories.

- **Alon et al. (2019)** – Introduced `code2vec`, which learns semantic representations of code via syntactic paths. This supports the project's AST- and parser-based approach to classifying parallelism intensity.

- **Liu & Weissman (2015)** – Developed Elastic Job Bundling for large-scale HPC environments. Their analysis underscores the growing demand for GPU-scale elasticity in compute workloads.

- **Allamanis et al. (2018)** – Demonstrated how deep learning models trained on AST structures can be used for source code understanding, reinforcing our use of syntax tree models (e.g., Tree-sitter) for parallel vs sequential detection.

- **Dyer et al. (2013)** – Developed the Boa language and curated millions of software repositories to support large-scale software mining, directly relevant to this project’s GitHub scraping and curation phase.

- **XGBoost Docs (2025)** – Emphasize GPU support and high performance on structured data, providing justification for its use in earnings prediction models within our pipeline.

- **Tree-sitter Documentation (2024)** – Confirms its lightweight and language-agnostic parsing capabilities, validating our approach to static code analysis across thousands of repositories.
- **CUDA C++ Programming Guide (NVIDIA, 2025)** – Provides a scalable parallel computing model (CUDA kernels, blocks, thread hierarchies) that forms the basis for recognizing and interpreting GPU-targeted code patterns in our repository mining.

---

## Methodology

### 1. Repository Mining
- Use GitHub API to collect and rank repositories tagged with CUDA, OpenCL, Deep Learning, and HPC.
- Filter by update recency and star count; extract metadata and language composition.

### 2. Parallelism Detection
- Parse source code using Tree-sitter, ASTs, and regex to isolate parallel code segments (e.g., CUDA kernels, multiprocessing calls).
- Compute parallelism ratio = parallel LOC / (parallel + sequential LOC).

### 3. Demand Mapping
- Classify projects:
  - **High parallelism** → Enterprise GPUs (e.g., H100, A100)
  - **Medium parallelism** → Consumer GPUs (e.g., RTX 4090, 4080)
  - **Low parallelism** → Entry-level GPUs
- Generate quarterly time series of project counts per GPU tier.

### 4. Supply Aggregation
- Scrape quarterly supply/production data from:
  - NVIDIA earnings reports
  - SEC EDGAR filings
  - Tech news sites (e.g., Tom’s Hardware, TechPowerUp)

### 5. EPS Modeling
- Train models (Linear Regression, XGBoost, LSTM) using:
  - Input: Demand growth rate, supply volume, demand-supply delta
  - Output: EPS
- Validate against historical EPS data.

### 6. Visualization
- Plot:
  - Parallelism trends over time
  - Projected demand vs supply
  - Predicted EPS vs actual EPS
  - Heatmaps of code structure by GPU class

---

## Tools & Libraries

- `Python`, `pandas`, `numpy`, `xgboost`, `matplotlib`, `seaborn`, `tree-sitter`, `hmmlearn`, `regex`
- Optional: `plotly`, `streamlit`, `scikit-learn`, `PyGitHub`

---

## Results Preview

- Trends show a strong rise in highly-parallel projects over recent quarters.
- Demand spikes for enterprise GPUs precede supply expansion announcements.
- EPS forecasting using demand-supply signals achieved high correlation with actuals.

---

## Potential Research Extensions

- Apply NLP to README files for project-level GPU model mentions.
- Introduce weighted scoring by stars/forks to refine demand signal strength.
- Extend model to AMD/Intel GPUs and generalize beyond CUDA to Vulkan/Metal.
