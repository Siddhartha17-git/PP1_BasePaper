# Base Paper Core Recommendation Engine Reproduction

Reproduction of the core recommendation engine from the IEEE Access (2025) paper:
> **"AI-Driven Personalized Learning and Remedial Recommendation Through Knowledge Concept-Centric Evaluation"**  
> *Pradeesh et al., IEEE Access, vol. 13, 2025.*  
> **DOI**: [10.1109/ACCESS.2025.3638427](https://doi.org/10.1109/ACCESS.2025.3638427)

---

## 📌 Executive Summary

This repository reproduces the **core AI recommendation pipeline** proposed in Pradeesh et al. (2025), focusing strictly on:
1. **Schema Inspection & Preprocessing**: Auto-mapping raw learner response logs to canonical schemas, filtering top concept spaces, and establishing reproducible learner-level train/test splits.
2. **Exercise-aware Knowledge Tracing (EKT)**: A deep recurrent model capturing multi-concept learner mastery across sequential exercise attempts.
3. **Outcome-Based Education (OBE) Knowledge Gap Engine**: Diagnostic engine categorizing learners into Beginner, Intermediate, and Expert tiers, and flagging actionable concept gaps.
4. **Pedagogical Remedial Recommendation Engine**: A structured five-section generator (Figure 2) paired with a multi-round post-remediation reassessment loop.

---

## 📁 Repository Structure

```
BaseExec/
├── config.yaml                # Global hyperparameters and path configurations
├── requirements.txt           # Python package requirements
├── README.md                  # Project overview and reproduction notes
├── run_core.py                # Single-command end-to-end pipeline orchestrator
├── inspect_data.py            # Dataset inspection and canonical column auto-mapper
├── prep_data.py               # Preprocessing, difficulty derivation, sequence chunking, split
├── ekt.py                     # EKT neural network, training loop, and get_knowledge_state API
├── gap_engine.py              # OBE mastery classifier, paper demo verification, gap detection
├── remedial.py                # 5-section remedial generator (Figure 2) & re-assessment loop
├── generate_figures.py        # Publication-quality figure generation
│
├── data/
│   ├── raw/                   # User-supplied raw benchmark dataset (skill_builder_data.csv)
│   └── processed/             # interactions.parquet, split.json, concepts.json
│
└── outputs/
    ├── tables/
    │   ├── ekt_metrics.csv                # Per-epoch train/test Loss, AUC, and MAE
    │   ├── paper_vs_reproduced_ekt.csv    # Paper-reported vs reproduced test metrics
    │   └── dataset_stats_comparison.csv   # Dataset stats beside paper's Table 1
    ├── figures/
    │   ├── metrics_curves.png             # Loss, AUC, MAE vs epoch (train dashed, test solid)
    │   ├── knowledge_state_heatmap.png    # Dynamic knowledge state heatmap across time
    │   └── mastery_radar_chart.png        # Polar radar chart of initial vs final concept mastery
    ├── models/
    │   └── ekt_checkpoint.pt              # PyTorch trained EKT model checkpoint
    ├── logs/
    │   └── column_map.json                # Canonical schema mapping configuration
    ├── gap_report.md                      # Detailed OBE gap diagnostic report
    ├── remedial_figure2.md                # Figure 2 structured remedial recommendation & loop log
    └── SUMMARY.md                         # Comprehensive reproduction summary & honest deviations
```

---

## ⚙️ Installation & Requirements

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

---

## 🚀 One-Command Execution

To execute the entire end-to-end pipeline (inspection, preprocessing, training, gap detection, remedial loop, figure generation, and verification):

```bash
python run_core.py
```

Individual modules can also be run independently:
```bash
python inspect_data.py       # Inspect raw dataset schema and save column mapping
python prep_data.py          # Clean, chunk sequences, derive difficulty, save split
python ekt.py                # Train 12-epoch EKT model and log performance metrics
python gap_engine.py         # Run paper demo vector & diagnose held-out learner gaps
python remedial.py           # Generate Figure 2 remedial recommendation & execute loop
python generate_figures.py   # Generate publication-quality figures
```

---

## 🔬 Core Components & Implementation Details

### 1. Data Inspection (`inspect_data.py`)
- Automatically detects dataset files in `data/raw/` (`.csv`, `.tsv`, `.parquet`, `.xlsx`), testing UTF-8 and Latin-1 encodings.
- Auto-maps fields to canonical names: `learner_id`, `order`, `question_id`, `concept_id`, `concept_name`, `correct`.
- Validates binary correctness (0/1) and detects absence of raw problem text.

### 2. Preprocessing & Splitting (`prep_data.py`)
- Deduplicates multi-skill records sharing the same order key per learner.
- Tracks the top $N=20$ concepts by frequency and generates continuous integer index mappings (`concepts.json`).
- Establishes a learner-level 75/25 seeded split (`random_state=42`) saved to `split.json`.
- Derives question difficulty strictly from the **train split** problem correct rates into terciles:
  - **Easy**: Upper tercile ($\ge 66.7\%$)
  - **Medium**: Middle tercile ($33.3\% - 66.7\%$)
  - **Hard**: Lower tercile ($\le 33.3\%$)

### 3. EKT Model Architecture (`ekt.py`)
- **Input Representation**: Combines exercise embedding ($d=100$) with concept-response interaction embedding ($d=100$) via a non-linear fusion layer.
- **Backbone**: LSTM with 100 hidden units and dropout $p=0.3$.
- **Mastery Output**: At each time step $t$, the hidden state $h_t$ is projected through a sigmoid layer producing an $N$-dimensional vector $y_t \in (0, 1)^N$, tracking mastery across all concepts simultaneously.
- **Optimization**: Binary Cross-Entropy loss on next-question correctness targeting $y_t[c_{t+1}]$, optimized with Adam ($lr=0.001$, batch size 32) over 12 epochs.
- **Evaluation**: Logs BCE Loss, ROC-AUC, and MAE for both train and test sets at every epoch.

### 4. OBE Knowledge Gap Engine (`gap_engine.py`)
- Implements the paper's Outcome-Based Education (OBE) three-tier classification:
  - **Beginner**: $\text{Mastery} < 50.0\%$ (Gap Flagged)
  - **Intermediate**: $50.0\% \le \text{Mastery} \le 70.0\%$ (Gap Flagged)
  - **Expert**: $\text{Mastery} > 70.0\%$ (Proficient / No Gap)
- Verified against the paper's demo vector `[0.82, 0.35, 0.76, 0.60, 0.91]`.
- Evaluates real held-out test learners from the saved EKT checkpoint.

### 5. Remedial Recommendation Generator (`remedial.py`)
- Constructs the paper's exact prompt requesting five structured sections:
  1. *Learning Objectives*
  2. *Recommended Topics to Revise*
  3. *Remedial Explanation*
  4. *Two Practice Activities* (Scaffolded & Independent)
  5. *Concept Gap Rationale*
- **Offline / Live Operation**: Deterministic offline pedagogical engine by default; optional live OpenAI API generation when `OPENAI_API_KEY` is present.
- **Post-Remediation Loop**: Re-assesses learner responses over up to 3 rounds, dynamically feeding outcomes back into EKT to update knowledge state $h_t$.

---

## 📊 Summary of Results & Deviations

### Dataset Statistics Beside Paper Table 1

| Metric | Base Paper (AMPLE LMS) | Reproduced Dataset (ASSISTments) |
|---|---|---|
| Concepts | 382 | 20 |
| Learners | 1,276 | 1,723 |
| Questions / Problems | N/A | 7,377 |
| Total Interactions | 359,681 | 154,244 |
| Avg Interactions / Learner | ~281 | 89.5 |
| Avg Correct Rate | 61.67% | 66.14% |

### EKT Test Performance

| Metric | Paper Reported EKT | Reproduced EKT (Final Epoch) | Reproduced EKT (Best Checkpoint) |
|---|---|---|---|
| **Test Loss (BCE)** | 0.3266 | 0.5495 | **0.5461** |
| **Test ROC-AUC** | 0.9334 | 0.7394 | **0.7405** |
| **Test MAE** | 0.1828 | 0.3537 | **0.3523** |

### Honest List of Deviations
1. **Proprietary vs Benchmark Data**: The original AMPLE LMS dataset is proprietary and unavailable. The pipeline runs strictly on the user-provided ASSISTments dataset.
2. **Trainable Question Embeddings**: No textual questions were present in the raw data; trainable question embeddings ($d=100$) were used in place of `all-MiniLM-L6-v2` text embeddings.
3. **Derived Difficulty**: Bloom's Taxonomy cognitive tags were unavailable; difficulty was empirically derived via terciles on train-split problem correct rates.
4. **Concept Space**: Constrained to the top 20 most frequent concepts to prevent extreme sequence sparsity.
5. **Absolute Performance**: Performance values naturally reflect the differing response dynamics and scale of the benchmark dataset while preserving the architecture and convergence trends.
