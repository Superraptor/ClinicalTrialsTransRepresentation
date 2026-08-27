# ClinicalTrials.gov Transgender & Gender-Diverse (TGD) Representation Pipeline

Data ingestion, natural language processing, unsupervised semantic clustering, and quantitative evaluation pipeline for investigating the historical emergence, categorization, and formal vs. informal (in)visibility of transgender and gender-diverse persons in [ClinicalTrials.gov](https://clinicaltrials.gov).

---

## 🎯 Research Focus

- **Historical Emergence & Normalization**: Longitudinal trajectory of TGD research (1999–present) normalized against platform growth (rate per 1,000 registered studies).
- **Formal vs. Informal Capture**: Disentangling structured metadata (MeSH terms, `genderBased`, `eligibility.sex`, keywords) from informal capture (free-text titles, summaries, descriptions, and eligibility criteria).
- **Unsupervised Semantic Clustering & Characterization**: Identifying latent research themes (e.g., HIV/MSM syndemic trials, gender-affirming endocrine/surgical care, oncology/cancer screening disparities, mental health/minority stress, and routine clinical trial inclusion/exclusion) using sentence embeddings, UMAP, and class-based TF-IDF (c-TF-IDF).
- **Structural Discordance & Ambiguity**: Systematic analysis of mismatches between structured eligibility constraints and free-text criteria.

---

## 🏗️ Architecture

```
ClinicalTrialsTransRepresentation/
├── config/
│   ├── lexicon.yaml                 # Multi-era TGD regex patterns, acronym guards, and negative filters
│   ├── mesh_terms.yaml              # Historical & contemporary MeSH headings (D063106, D014164, etc.)
│   └── pipeline_config.yaml         # API endpoints, batching, clustering parameters, and data paths
├── data/
│   ├── raw/                         # Cached API responses, consolidated JSON, baseline counts
│   ├── processed/                   # Flattened Parquet / CSV datasets (studies, rules, matches, clusters)
│   └── manual_validation/          # Stratified sample for human-in-the-loop validation
├── reports/
│   ├── figures/                     # Publication-ready static PNGs and interactive plots
│   ├── tables/                      # Formatted Markdown and CSV manuscript tables
│   └── summary_metrics.json         # High-level statistical summaries
├── src/
│   ├── ingestion/                   # API v2 client, query sweep, and denominator tracker
│   ├── parsing/                     # JSON flattener, criteria segmenter, and term extractor
│   ├── classification/              # Unsupervised clustering, c-TF-IDF, concordance matrix, benchmark taxonomy
│   ├── evaluation/                  # Growth rates, odds ratios, temporal trajectories, manual validation
│   └── visualization/               # Publication figures and summary tables
├── tests/                           # Pytest unit and integration test suite
├── run_pipeline.py                  # End-to-end CLI orchestrator
└── requirements.txt
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline in Demo Mode (Fast Sample Execution)
Runs a focused multi-term query sweep, parses studies, runs unsupervised clustering, extracts c-TF-IDF characterizations, computes metrics, and generates publication figures:
```bash
python run_pipeline.py --mode demo
```

### 3. Run Full End-to-End Ingestion & Analysis
```bash
python run_pipeline.py --mode all
```

### 4. Run Specific Pipeline Stages
- **Fetch Denominator Baseline (1999–2026)**:
  ```bash
  python run_pipeline.py --mode baseline
  ```
- **Parse & Extract Terms from Cached Data**:
  ```bash
  python run_pipeline.py --mode parse
  ```
- **Run Unsupervised Clustering & Characterization**:
  ```bash
  python run_pipeline.py --mode cluster
  ```
- **Generate Tables & Metrics**:
  ```bash
  python run_pipeline.py --mode evaluate
  ```
- **Generate Figures**:
  ```bash
  python run_pipeline.py --mode visualize
  ```

---

## 📊 Generated Manuscript Outputs

### Tables (`reports/tables/`)
- **Table 1 (`table1_cohort_summary.md`)**: Total TGD studies, temporal span, formal vs. informal capture breakdown, NIH vs. Industry funding proportions, study types, and results reporting rates.
- **Table 2 (`table2_unsupervised_clusters.md`)**: Thematic cluster profiles, descriptive c-TF-IDF labels, study volumes, mean years, NIH funding %, formal capture %, and exemplar NCT IDs.
- **Table 3 (`table3_structural_discordance.md`)**: Structural discordance breakdown between formal eligibility constraints and free-text criteria.

### Figures (`reports/figures/`)
- **Figure 1**: Longitudinal Emergence & Normalization Rate per 1,000 registered studies (1999–present).
- **Figure 2**: Stacked Formal vs. Informal Capture Over Time.
- **Figure 3**: Epistemic Trajectory of Terminology (Psychiatric vs. Diagnostic vs. Affirmative vs. Medical).
- **Figure 4**: 2D UMAP Semantic Projection of Unsupervised Thematic Clusters.
- **Figure 5**: Cluster Trajectory Shifts Over Time.

---

## 🧪 Testing
```bash
python -m pytest -v
```
