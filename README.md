# Recommender Regime Pipeline

This repository contains the experiment code and cleaned result tables used for the MSc thesis:

**Exploring Performance Variability in Recommender Systems: The Role of Sparsity and Dataset Characteristics**

The pipeline prepares controlled user–item interaction regimes, trains RecBole models, and evaluates Top-K recommendation quality with an external sampled-candidate evaluator.

## Submission artifact

The final submitted state should be identified by a fixed commit and release tag rather than by the mutable `main` branch.

Planned submission tag:

```text
v1.0-thesis-submission
```

Create the tag only after the final README, code, result CSVs, and SHA256 manifest have been committed and checked.

## Repository contents

| Path | Purpose |
|---|---|
| `common.py` | Shared configuration, input normalization, splitting, candidate sampling, ranking metrics, and artifact paths. |
| `prepare_variants.py` | Builds fixed-users, fixed-items, k-core, random-drop, tail-drop stress, size-matched k-core, and add-back variants. |
| `recbole_models.py` | Trains and evaluates BPR, ItemKNN, LightGCN, NeuMF, and MultiVAE through RecBole. |
| `preprocessing_lastfm.py` | Converts Last.fm play-count data to implicit interactions. |
| `preprocessing_online_retail.py` | Converts Online Retail transactions to implicit interactions. |
| `preprocessing_steam.py` | Converts Steam review records to implicit interactions. |
| `requirements.txt` | Python packages used by the recorded experiment environment. |
| `results/main_results.csv` | Main regime-family aggregates. |
| `results/results_kcore_sizematch.csv` | Size-matched k-core control. |
| `results/results_aggressive_taildrop.csv` | Auxiliary fixed-users tail-drop stress test. |
| `results/addback_results.csv` | KC50 add-back diagnostic used in Appendix A.10. |
| `artifact_manifest_sha256.txt` | SHA256 hashes for the final submitted files. |

Raw datasets, materialized variants, full RecBole run directories, model checkpoints, saved recommendation lists, and LaTeX sources are not included.

## Experiment overview

The thesis compares several ways of changing interaction density and retained support:

- fixed-users item head/tail filtering;
- fixed-items user head/tail filtering;
- k-core filtering with `k = 5, 10, 50`;
- random interaction drop under fixed support;
- size-matched k-core controls;
- a KC50 add-back diagnostic that restores selected KC5-minus-KC50 interactions.

The model portfolio is:

```text
BPR, ItemKNN, LightGCN, NeuMF, MultiVAE
```

The main ranking metric is `NDCG@10`. The code also reports `Precision@10`, `Recall@10`, and `MRR@10`; optional beyond-accuracy metrics can be enabled separately.

## Expected interaction format

The main scripts expect an input table that can be normalized to:

```text
user_id,item_id,rating[,timestamp]
```

`timestamp` is optional. Timestamped datasets can use chronological leave-one-out; otherwise the split falls back to random leave-one-out under the configured seed.

Explicit ratings can be converted to implicit positives. The default rule is:

```text
rating >= ceil(0.8 * RATING_MAX)
```

For a 1–5 scale, this retains ratings 4 and 5.

## Output layout

Generated outputs are written under:

```text
$EXPORT_BASE/$DATASET_NAME/$RUN_ID/
```

Typical subdirectories are:

```text
variants/    materialized ratings_*.csv files
splits/      train.csv, test.csv, and metadata
artifacts/   RecBole input files, model states, metrics, and optional Top-K files
results/     per-seed and aggregated result CSVs
meta/        run-level metadata
```

The repository-level `results/` directory contains only cleaned aggregate tables copied from completed runs.

## Installation

Create a Python environment and install the recorded dependencies:

```bash
pip install -r requirements.txt
```

The recorded environment used Python 3.9 and RecBole 1.2.1. GPU execution used PyTorch 2.6.0 with CUDA 12.4; CPU execution is also supported for code inspection and smaller runs.

## Main configuration

The scripts are controlled through environment variables. The most important settings are:

| Variable | Purpose |
|---|---|
| `INPUT_RATINGS_PATH` | Input interaction file. |
| `EXPORT_BASE` | Base output directory. |
| `DATASET_NAME` | Dataset name used in output paths. |
| `RUN_ID` | Shared identifier for preparation and model execution. |
| `SEED` | Main construction seed. |
| `SEEDS` | Comma-separated model-run seeds, e.g. `42,43,44`. |
| `TOPK` | Ranking cutoff; thesis value: `10`. |
| `NEGATIVE_CANDIDATE_SAMPLE` | Sampled negatives per evaluable user; thesis value: `1000`. |
| `RECBOLE_DEVICE` | `cpu` or a CUDA device such as `cuda:0`. |
| `RUN_MODEL_LIST` | Comma-separated model subset. |

## Main run

Use the same `RUN_ID` for variant preparation and model execution.

```bash
export RUN_ID=my_run_$(date +%Y%m%d_%H%M%S)
export INPUT_RATINGS_PATH=/path/to/ratings.csv
export EXPORT_BASE=/path/to/exports
export DATASET_NAME=my_dataset
export SEED=42
export SEEDS="42,43,44"
export TOPK=10
export NEGATIVE_CANDIDATE_SAMPLE=1000
```

Prepare variants and splits:

```bash
export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0
python -u prepare_variants.py
```

Train and evaluate models:

```bash
export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1
export RUN_MODEL_LIST="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"
python -u recbole_models.py
```

## Variant families

Common preparation switches include:

| Family | Main settings |
|---|---|
| Fixed-users item filtering | `PREP_ITEM_VARIANTS=1` |
| Fixed-items user filtering | `PREP_USER_VARIANTS=1` |
| k-core | `PREP_KCORE_VARIANTS=1`, `KCORE_KS="5,10,50"` |
| Random drop | `PREP_RANDOM_DROP_VARIANTS=1`, `RANDDROP_PCTS="0.2,0.4,0.6"` |
| Tail-drop stress | `PREP_ONLY_TAILDROP_STRESS=1`, `TAILDROP_STRESS_PCTS="0.4,0.6,0.8"` |
| Size-matched k-core | `PREP_ONLY_KCORE_SIZE_MATCH=1`, `KCORE_SIZE_MATCH_TARGET=50` |
| Add-back | `PREP_ONLY_ADDBACK_VARIANTS=1`, `ADDBACK_SOURCE_K=5`, `ADDBACK_TARGET_K=50` |

## KC50 add-back diagnostic

The add-back experiment starts from KC50 and restores selected interactions that are present in KC5 but absent from KC50. The submitted percentage grid is:

```text
10%, 25%, 50%, 100%
```

Typical preparation settings:

```bash
export PREP_ONLY_ADDBACK_VARIANTS=1
export PREP_ADDBACK_VARIANTS=1
export ADDBACK_SOURCE_K=5
export ADDBACK_TARGET_K=50
export ADDBACK_PCTS="0.1,0.25,0.5,1.0"
export ADDBACK_SEEDS="42"
export ADDBACK_SAVE_TARGET_PLAIN=1
export ADDBACK_COMMON_TEST_FROM_TARGET=1
```

Typical model settings:

```bash
export RUN_ONLY_ADDBACK_VARIANTS=1
export RUN_ADDBACK_VARIANTS=1
export SEEDS="42,43,44"
export RUN_MODEL_LIST="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"
```

`ADDBACK_SEEDS` controls construction of the add-back variants. `SEEDS` controls repeated model training and evaluation. These are different sources of randomness and should not be conflated.

The add-back diagnostic is descriptive rather than causal. Restoring interactions can also restore many users and items, so the experiment changes retained support as well as edge count. It does not include matched random, degree-matched, or popularity-matched edge-addition controls.

## Submitted result files

The four CSV files under `results/` are the numerical sources used by the thesis:

```text
results/main_results.csv
results/results_kcore_sizematch.csv
results/results_aggressive_taildrop.csv
results/addback_results.csv
```

`addback_results.csv` contains the final refreshed Last.fm and MovieLens common-test12 exports together with the Online Retail and Steam add-back runs.

## Artifact manifest

After the final files are fixed, generate a SHA256 manifest. On PowerShell:

```powershell
$files = @(
  "README.md",
  "common.py",
  "prepare_variants.py",
  "recbole_models.py",
  "preprocessing_lastfm.py",
  "preprocessing_online_retail.py",
  "preprocessing_steam.py",
  "requirements.txt",
  "results/main_results.csv",
  "results/results_kcore_sizematch.csv",
  "results/results_aggressive_taildrop.csv",
  "results/addback_results.csv"
)

$manifest = foreach ($file in $files) {
  $hash = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
  "$hash  $file"
}

$manifest | Set-Content -Encoding ascii artifact_manifest_sha256.txt
```

Check the final commit identifier with:

```bash
git rev-parse HEAD
```

## Submission tag and release

After the final cleanup commit has been pushed:

```bash
git tag -a v1.0-thesis-submission -m "MSc thesis submission artifact"
git push origin v1.0-thesis-submission
```

Create the GitHub release from the same tag. The thesis should cite the fixed tag or commit, not the mutable `main` branch.
