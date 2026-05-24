# FT-Transformer Credit-Risk Study

End-to-end credit-risk modelling on two real-world binary-classification
datasets — **Lending Club** and **Home Credit Default Risk** — comparing
classical baselines (Logistic Regression, Random Forest, XGBoost) against
tabular transformer architectures (**FT-Transformer**, **SAINT**,
**ExcelFormer**), with a range of architectural and preprocessing ablations
on top of FT-Transformer.

The project is organised so every model in both datasets runs through the
same preprocessing pipeline, the same evaluation metric suite, and the same
artifact-tracking layer.

---

## Repository layout

```
ft-transformer-credit-risk-study/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/                            # shared utilities — used by baseline notebooks
│   ├── __init__.py
│   ├── datasets.py                 #   raw loaders + dataset-specific cleaning
│   ├── preprocessing.py            #   leakage-free 64/16/20 split + impute/scale/encode
│   ├── evaluation.py               #   shared metric suite (AUC, Gini, KS, AUCPR, top-decile)
│   ├── interpretation.py           #   SHAP helpers for tree models
│   └── tracking.py                 #   ExperimentLogger, timing, artifact persistence
│
├── notebooks/
│   ├── lending_club/
│   │   ├── baseline/
│   │   │   └── 01_lending_club_baselines.ipynb     # LR / RF / XGB on Lending Club
│   │   ├── 00_xgb_baseline_and_ensemble.ipynb      # XGB Optuna + FTT×XGB blend
│   │   ├── 01_ftt_vanilla_hyperopt.ipynb           # Vanilla FTT hyperopt (reference run)
│   │   ├── 02_ftt_swiglu_variants.ipynb            # Four SwiGLU placement variants
│   │   ├── 03_ftt_custom_embeddings.ipynb          # Linear+PLE / MLP / Cross-SwiGLU / Periodic
│   │   ├── 04_ftt_optimizer_and_loss_variants.ipynb# Lion optimizer + Pairwise AUC loss
│   │   ├── 05_ftt_preprocessing_quantile.ipynb     # QuantileTransformer preprocessing
│   │   ├── 06_ftt_preprocessing_pca_whitening.ipynb# log1p + PCA whitening preprocessing
│   │   ├── 07_xgb_rotation_features.ipynb          # Random rotation of numerical features
│   │   ├── 08_saint_baseline.ipynb                 # SAINT
│   │   └── 09_excelformer_baseline.ipynb           # ExcelFormer
│   │
│   └── home_credit/
│       ├── baseline/
│       │   └── 01_home_credit_baselines.ipynb      # LR / RF / XGB on Home Credit
│       ├── 00_ftt_architecture_variants.ipynb      # Ten FTT modifications benchmarked side-by-side
│       ├── 01_ftt_preprocessing_pca.ipynb          # PCA whitening for XGB + FTT
│       ├── 02_saint_baseline.ipynb                 # SAINT
│       └── 03_excelformer_baseline.ipynb           # ExcelFormer
│
├── data/                           # input files (git-ignored)
└── artifacts/                      # per-run outputs (git-ignored)
    ├── lending_club/{models,results,figures}/
    └── home_credit/{models,results,figures}/
```

The two notebook *types* differ on purpose:

- **Baseline notebooks** (`notebooks/<dataset>/baseline/`) are the reference
  pipeline. They import from `src/` so preprocessing, scoring and artifact
  paths are uniform across both datasets, and write all per-run artifacts
  through `ExperimentLogger`.
- **FT-Transformer / SAINT / ExcelFormer notebooks** (`notebooks/<dataset>/`)
  are **self-contained** and each one inlines its own
  preprocessing block, defines its own model class (the modifications **are**
  the experiment), and writes its own checkpoints / JSON / figures to
  `artifacts/<dataset>/{models,results,figures}/`. They do **not** import
  `src/`.

---

## Datasets

| Dataset             | Source                                                                                            | Target                                  |
|---------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------|
| Lending Club        | Public Lending Club loan dataset (Kaggle: *Lending Club Accepted Loans 2007-2018*)                | `target_binary` (1 = charged-off/default) |
| Home Credit         | [Home Credit Default Risk competition](https://www.kaggle.com/c/home-credit-default-risk)         | `TARGET` (1 = default)                  |

Input files are not shipped with the repository. Drop your local copies
under `data/`.

---

## Running the notebooks

Every notebook follows the same path-resolution convention:

```python
# Uncomment on Colab:
# from google.colab import drive
# drive.mount("/content/drive")
# DRIVE_ROOT = "/content/drive/MyDrive/ft-transformer-credit-risk-study"

_BASE = globals().get("DRIVE_ROOT", "..")
DATA_PATH      = f"{_BASE}/data/<filename>"
ARTIFACTS_DIR  = Path(f"{_BASE}/artifacts/<dataset>")
```

- **Local Jupyter**: `_BASE` defaults to `..` (notebooks are one level deep
  under `notebooks/<dataset>/`, or two under `notebooks/<dataset>/baseline/`
  — adjust accordingly).
- **Colab**: uncomment the Drive-mount block and set `DRIVE_ROOT` to your
  Drive folder. The next cell auto-routes through it.

### Installation

```bash
git clone <this-repo>
cd ft-transformer-credit-risk-study
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

A CUDA-capable GPU is strongly recommended for the transformer notebooks
(otherwise expect hours per Optuna trial). The baseline LR / RF / XGB
notebooks and the pure-XGB experiments run fine on CPU; the XGB hyperopt
cells use `device='cuda'` if available and fall back to CPU otherwise.

### `DEV_MODE` (FT-Transformer notebooks only)

Each transformer training notebook has a `DEV_MODE = False` flag near the
top of the setup section. Set it to `True` for a fast smoke test:

- subsamples each data split to ~5 000 rows,
- forces `n_epochs = 2`,
- forces Optuna `n_trials = 1`.

When `DEV_MODE = False` (the default), every constant resolves to its
original literal value, so the run is behaviourally identical to the source
experiment.

`DEV_MODE` is **omitted** from the pure-XGBoost notebooks where a single
tuned `.fit()` is already fast and a switch would just add noise.

---

## Architectures touched

- **Logistic Regression**, **Random Forest**, **XGBoost** — classical
  baselines, in `notebooks/<dataset>/baseline/`.
- **FT-Transformer** (Gorishniy et al., 2021), upstream reference is
  `rtdl_revisiting_models.FTTransformer`. Each variant notebook keeps its own
  modified copy of the class — the modifications are the experiment.
- **SAINT** — Self-Attention and Intersample Attention Transformer
  (Somepalli et al., 2021).
- **ExcelFormer** — tabular transformer with semi-permeable attention and
  feat-mix / hidden-mix augmentation (Chen et al., 2023).

---

## License

MIT — see `LICENSE`.
