# Recommender Regime Pipeline

Pipeline for preparing recommendation datasets and running RecBole baselines on the prepared variants.

The project has two main steps:

1. `prepare_variants.py` creates the dataset variants and train/test splits.
2. `recbole_models.py` trains RecBole models and evaluates them with the shared external evaluator.

Most shared code is in `common.py`. The three `preprocessing_*.py` files are small converters for LastFM, Online Retail, and Steam data.

This repository snapshot contains the code, aggregate result CSVs, and the thesis PDF. It intentionally does not contain LaTeX sources, raw datasets, generated variant CSVs, RecBole artifacts, saved model checkpoints, or full recommendation-list outputs.

## Repository contents

| File | What it is used for |
|---|---|
| `common.py` | Config handling, path setup, splitting, sampled candidates, metrics, and evaluation helpers. |
| `prepare_variants.py` | Builds raw/aligned variants, sparse/dense variants, k-core variants, random-drop variants, tail-drop stress variants, size-matched k-core controls, and add-back variants. |
| `recbole_models.py` | Runs BPR, ItemKNN, LightGCN, NeuMF, and MultiVAE through RecBole. |
| `preprocessing_lastfm.py` | Converts LastFM playcount TSV data to the expected interaction format. |
| `preprocessing_online_retail.py` | Converts Online Retail transactions to implicit interactions. |
| `preprocessing_steam.py` | Converts Steam review logs to implicit interactions. |
| `requirements.txt` | Python package versions used for the recorded run. |
| `docs/thesis.pdf` | Thesis PDF included for submission/reference. LaTeX sources are intentionally excluded. |
| `results/*.csv` | Aggregate result CSVs used by the thesis: main results, aggressive tail-drop, size-matched k-core, and add-back. |
| `supplementary/reproducibility/` | Manifest helper and SHA256 manifest for the submitted repository snapshot. |

## Input data

The main scripts expect a CSV that can be normalized to this format:

```text
user_id,item_id,rating[,timestamp]
```

`timestamp` is optional. If it is present, the split can use chronological leave-one-out. Without it, the code uses random leave-one-out with the configured seed.

For explicit ratings, the default binarization keeps ratings above this threshold:

```text
rating >= ceil(0.8 * RATING_MAX)
```

For a 1-5 scale this means ratings 4 and 5 are treated as positive interactions.

## Output structure

Pipeline outputs are written under:

```text
$EXPORT_BASE/$DATASET_NAME/$RUN_ID/
```

The repository-level `results/` directory is different: it contains the submitted aggregate CSVs copied from completed runs.

Main folders:

```text
variants/       ratings_<variant>.csv files
splits/         train.csv, test.csv, and meta.json for each variant
artifacts/      RecBole data files, saved models, metrics, and optional top-k files
results/        result CSVs aggregated by seed and by variant/model
meta/           run metadata and size-match metadata
```

Use the same `RUN_ID` for the preparation step and the model step. If the two commands are run in different terminals, export the same value manually.

## Installation

Create a Python environment and install the recorded dependencies:

```bash
pip install -r requirements.txt
```

For a quick local check without the exact CUDA build, the important packages are:

```bash
pip install numpy pandas scipy joblib torch recbole scikit-learn scikit-surprise tqdm PyYAML
```

`pyarrow` is optional. It is only needed when parquet output is used.

## Main configuration

The scripts are configured through environment variables. These are the ones that usually matter first:

| Variable | Default | Notes |
|---|---:|---|
| `INPUT_RATINGS_PATH` | `~/datasets/movielens/rating.csv` | Input interaction file. |
| `EXPORT_BASE` | `~/exports_sparsity_new` | Base output directory. |
| `DATASET_NAME` | inferred | Name used in the output path. |
| `RUN_ID` | timestamp | Set this yourself if the run needs to be reproduced. |
| `SEED` | `42` | Main random seed. |
| `SEEDS` | `SEED` | Comma-separated training seeds, for example `42,43,44`. |
| `TOPK` | `10` | Cutoff for ranking metrics. |
| `NEGATIVE_CANDIDATE_SAMPLE` | `1000` | Number of sampled negatives per test user. Use `0` only when full-catalog ranking is feasible. |
| `BINARIZE_FROM_EXPLICIT` | dataset-dependent | Enables explicit-to-implicit conversion. |
| `BINARY_THRESHOLD_FRACTION` | `0.8` | Rating threshold fraction. |
| `RATING_MIN`, `RATING_MAX` | `1`, `5` | Explicit rating scale. |

## Basic run

### 1. Set the shared run id

```bash
export RUN_ID=my_run_$(date +%Y%m%d_%H%M%S)
```

### 2. Prepare variants and splits

```bash
export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0

python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"
```

### 3. Train and evaluate models

```bash
export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1

python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

## Variant preparation options

The preparation script can build several variant families. The default setup builds the main fixed-users and fixed-items variants, then creates common train/test split files for each generated variant.

Common switches:

| Variable | Default | Notes |
|---|---:|---|
| `PREP_ITEM_VARIANTS` | `1` | Builds `raw_aligned`, `sparse`, and `dense`. |
| `PREP_USER_VARIANTS` | `1` | Builds `raw_aligned_fixitems`, `user_sparse`, and `user_dense`. |
| `SAVE_RAW_FULL` | `0` | Saves the normalized full input as `ratings_raw_full.csv`. |
| `TARGET_USERS` | `20000` | Target user count for fixed-support regimes when possible. |
| `TARGET_ITEMS` | `5000` | Target item count for fixed-items and random-drop regimes when possible. |
| `ITEM_HEAD_DROP_PCT` | `0.10` | Fraction of most popular items removed for item head-drop. |
| `ITEM_TAIL_DROP_PCT` | `0.40` | Fraction of least popular items removed for item tail-drop. |
| `USER_HEAD_DROP_PCT` | `0.10` | Fraction of most active users removed for user head-drop. |
| `USER_TAIL_DROP_PCT` | `0.40` | Fraction of least active users removed for user tail-drop. |
| `PREP_COMMON_SPLITS` | `1` | Creates split files after building variants. |
| `COMMON_SPLIT_MODE` | `auto` | `auto`, `chrono_loo`, or `random_loo`. |
| `FORCE_COMMON_SPLIT_OVERWRITE` | `0` | Rewrites existing split files. |
| `FORCE_EXPORT_OVERWRITE` | `0` | Rewrites existing variant CSV files. |

Optional variant families:

| Variant family | Main switches |
|---|---|
| k-core | `PREP_KCORE_VARIANTS=1`, `KCORE_KS="5,10,50"`, `KCORE_SAVE_PLAIN=1` |
| random drop | `PREP_RANDOM_DROP_VARIANTS=1`, `RANDDROP_BASE=raw`, `RANDDROP_PCTS="0.2,0.4,0.6"` |
| tail-drop stress | `PREP_ONLY_TAILDROP_STRESS=1`, `TAILDROP_STRESS_PCTS="0.4,0.6,0.8"` |
| size-matched k-core | `PREP_ONLY_KCORE_SIZE_MATCH=1`, `KCORE_SIZE_MATCH_TARGET=50`, `KCORE_SIZE_MATCH_BASE_KS="5,10"` |
| k-core add-back | `PREP_ONLY_ADDBACK_VARIANTS=1`, `ADDBACK_SOURCE_K=5`, `ADDBACK_TARGET_K=50`, `ADDBACK_PCTS="0.1,0.25,0.5,1.0"`, `ADDBACK_SEEDS="42"` for the submitted one-variant-seed diagnostic |

A few details are easy to mix up:

- Tail-drop stress variants use one fixed common user set across `td40`, `td60`, and `td80`.
- Size-matched k-core controls match only the user and item counts of the target k-core. They do not reuse the target user set.
- Add-back is the family where the target test set and fixed negative-candidate file can be shared across variants.
- In the submitted add-back diagnostic, add-back variants are generated with one construction seed (`ADDBACK_SEEDS="42"`). Model training/evaluation is still repeated with three model-run seeds through `SEEDS="42,43,44"`. If a variant filename contains `_seed42`, that suffix refers to the add-back construction seed, not to the RecBole model-run seed.

## Model options

`recbole_models.py` discovers the selected variant files and runs the selected models. Reported metrics come from the external evaluator in this repository, not from RecBole's internal validation output.

| Variable | Default | Notes |
|---|---:|---|
| `RUN_MODEL_LIST` | all | Comma-separated subset, for example `BPR,LightGCN`. |
| `RUN_ITEM_VARIANTS` | `1` | Includes `raw_aligned`, `sparse`, and `dense`. |
| `RUN_USER_VARIANTS` | `1` | Includes `raw_aligned_fixitems`, `user_sparse`, and `user_dense`. |
| `RUN_KCORE_VARIANTS` | `1` | Includes k-core variants. |
| `RUN_KCORE_PLAIN_ONLY` | `1` | Uses only `ratings_kcore*_plain.csv` for k-core runs. |
| `RUN_RANDDROP_VARIANTS` | `0` | Includes random-drop variants. |
| `RUN_TAILDROP_STRESS_VARIANTS` | `0` | Includes `ratings_td*.csv`. |
| `RUN_ADDBACK_VARIANTS` | `0` | Includes add-back variants. |
| `RUN_VARIANT_GLOB` | empty | Optional filter for variant file names. |
| `RUN_TAG` | empty | Suffix for result CSV names. |
| `RECBOLE_DEVICE` | `cpu` | Example: `cuda:0`. |
| `RECBOLE_EPOCHS` | `20` | Training epochs. |
| `RECBOLE_TRAIN_BATCH_SIZE` | `1024` | Default train batch size. |
| `RECBOLE_EVAL_BATCH_SIZE` | `2048` | Default evaluation batch size. |
| `SKIP_FIT_VALIDATION` | `1` | Skips normal RecBole validation. |
| `REUSE_ARTIFACTS` | `1` | Reuses existing model metrics where available. |
| `SAVE_RECS` | `1` | Saves top-k recommendation files. Needed for `EVAL_ONLY=1`. |
| `EVAL_ONLY` | `0` | Recomputes metrics from saved top-k files without training. |
| `COMPUTE_BEYOND_ACCURACY` | `0` | Enables extra beyond-accuracy metrics. |
| `COMPUTE_ILS` | `0` | Enables ILS/diversity calculation. This can be slow. |

## Example: MovieLens main run

```bash
export EXPORT_BASE=~/projects/mrs_movielens/exports
export DATASET_NAME=movielens
export INPUT_RATINGS_PATH=/path/to/ratings_ml20m.csv
export RUN_ID=ml20m_main_$(date +%Y%m%d_%H%M%S)
export RUN_TAG=ml20m_main

export BINARIZE_FROM_EXPLICIT=1
export BINARY_THRESHOLD_FRACTION=0.8
export RATING_MIN=1
export RATING_MAX=5

export SEED=42
export SEEDS="42,43,44"
export TOPK=10
export NEGATIVE_CANDIDATE_SAMPLE=1000

export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0
export SAVE_RAW_FULL=0
export PREP_ITEM_VARIANTS=1
export PREP_USER_VARIANTS=1
export PREP_KCORE_VARIANTS=1
export KCORE_KS="5,10,50"
export KCORE_SAVE_PLAIN=1
export PREP_RANDOM_DROP_VARIANTS=1
export RANDDROP_PCTS="0.2,0.4,0.6"
export PREP_COMMON_SPLITS=1
export COMMON_SPLIT_MODE=auto
export COMMON_SPLIT_TIME_COL=timestamp
export COMMON_SPLIT_SEED=42
export FORCE_COMMON_SPLIT_OVERWRITE=1

python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"

export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1
export CUDA_VISIBLE_DEVICES=0
export RECBOLE_DEVICE=cuda:0
export RUN_MODEL_LIST="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"
export RUN_ITEM_VARIANTS=1
export RUN_USER_VARIANTS=1
export RUN_KCORE_VARIANTS=1
export RUN_KCORE_PLAIN_ONLY=1
export RUN_RANDDROP_VARIANTS=1
export RUN_ADDBACK_VARIANTS=0
export REUSE_ARTIFACTS=1
export SKIP_FIT_VALIDATION=1
export RECBOLE_EPOCHS=20

python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

## Example: tail-drop stress variants

This side run only builds `td40`, `td60`, and `td80` with a shared user set.

```bash
export EXPORT_BASE=~/projects/mrs_movielens/exports
export DATASET_NAME=movielens
export INPUT_RATINGS_PATH=/path/to/ratings_ml20m.csv
export RUN_ID=ml20m_taildrop_stress_$(date +%Y%m%d_%H%M%S)
export RUN_TAG=ml20m_taildrop_stress

export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0
export PREP_ONLY_TAILDROP_STRESS=1
export TAILDROP_STRESS_PCTS="0.4,0.6,0.8"
export TARGET_USERS=20000
export PREP_COMMON_SPLITS=1
export COMMON_SPLIT_FILE_GLOB="ratings_td*.csv"
export SPARSITY_REPORT_FILE_GLOB="ratings_td*.csv"
export FORCE_COMMON_SPLIT_OVERWRITE=1

python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"

export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1
export RUN_ONLY_TAILDROP_STRESS=1
export RUN_MODEL_LIST="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"

python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

## Example: size-matched k-core control

This matches KC5 and KC10 to the KC50 user/item counts. The split is still variant-specific leave-one-out.

```bash
export EXPORT_BASE=~/projects/mrs_movielens/exports
export DATASET_NAME=movielens
export INPUT_RATINGS_PATH=/path/to/ratings_ml20m.csv
export RUN_ID=ml20m_kcore_sizematch_$(date +%Y%m%d_%H%M%S)
export RUN_TAG=ml20m_kcore_sizematch

export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0
export PREP_ONLY_KCORE_SIZE_MATCH=1
export KCORE_SIZE_MATCH_TARGET=50
export KCORE_SIZE_MATCH_BASE_KS="5,10"
export KCORE_SIZE_MATCH_SEEDS="42,43,44"
export KCORE_SIZE_MATCH_INCLUDE_TARGET=1
export PREP_COMMON_SPLITS=1
export COMMON_SPLIT_FILE_GLOB="ratings_kcore*_sizematch_kcore50_seed*.csv,ratings_kcore50_plain.csv"
export SPARSITY_REPORT_FILE_GLOB="ratings_kcore*_sizematch_kcore50_seed*.csv,ratings_kcore50_plain.csv"
export FORCE_COMMON_SPLIT_OVERWRITE=1

python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"

export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1
export RUN_ONLY_KCORE_SIZE_MATCHED_VARIANTS=1
export KCORE_SIZE_MATCH_TARGET=50
export RUN_KCORE_SIZE_MATCH_TARGET=1
export RUN_MODEL_LIST="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"

python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

## Example: k-core add-back with fixed evaluation

```bash
export EXPORT_BASE=~/projects/mrs_movielens/exports
export DATASET_NAME=movielens
export INPUT_RATINGS_PATH=/path/to/ratings_ml20m.csv
export RUN_ID=ml20m_addback_k50_from_k5_$(date +%Y%m%d_%H%M%S)
export RUN_TAG=ml20m_addback_k50_from_k5

export BINARIZE_FROM_EXPLICIT=1
export BINARY_THRESHOLD_FRACTION=0.8
export RATING_MIN=1
export RATING_MAX=5
export SEED=42
export SEEDS="42,43,44"
export NEGATIVE_CANDIDATE_SAMPLE=1000

export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0
export PREP_ITEM_VARIANTS=0
export PREP_USER_VARIANTS=0
export PREP_KCORE_VARIANTS=0
export PREP_RANDOM_DROP_VARIANTS=0
export PREP_ONLY_ADDBACK_VARIANTS=1
export PREP_ADDBACK_VARIANTS=1
export ADDBACK_SOURCE_K=5
export ADDBACK_TARGET_K=50
export ADDBACK_PCTS="0.1,0.25,0.5,1.0"
export ADDBACK_SEEDS="42"          # one add-back construction seed for variants
export ADDBACK_SAVE_TARGET_PLAIN=1
export ADDBACK_COMMON_TEST_FROM_TARGET=1
export PREP_COMMON_SPLITS=1
export COMMON_SPLIT_MODE=auto
export COMMON_SPLIT_SEED=42
export FORCE_COMMON_SPLIT_OVERWRITE=1

python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"

export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1
export RUN_ONLY_ADDBACK_VARIANTS=1
export RUN_ADDBACK_VARIANTS=1
export RUN_VARIANT_GLOB="ratings_kcore50_plain.csv,ratings_kcore50_addback_from_kcore5_p*_seed42.csv"
export SEEDS="42,43,44"              # three model-run seeds on the same variants
export RECBOLE_DEVICE=cuda:0
export RUN_MODEL_LIST="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"

python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

## Dataset converters

### LastFM

```bash
python preprocessing_lastfm.py --in inter.txt --out ratings.csv
```

### Online Retail

```bash
python preprocessing_online_retail.py --in OnlineRetail.csv --out ratings_raw_full.csv
```

Use `--session-as-user` only when each invoice should be treated as a separate user.

### Steam

Check the first records:

```bash
python preprocessing_steam.py --in steam_reviews.json --out rating.csv --inspect
```

Convert the full file:

```bash
python preprocessing_steam.py --in steam_reviews.json --out rating.csv --db steam_reviews.sqlite --batch-size 5000
```

## Submitted result files

The repository includes these aggregate CSVs under `results/`:

| File | Contents |
|---|---|
| `main_results.csv` | Main regime-family aggregates used in the thesis. |
| `results_aggressive_taildrop.csv` | Auxiliary fixed-users tail-drop stress test. |
| `results_kcore_sizematch.csv` | Size-matched k-core control results. |
| `addback_results.csv` | Appendix add-back diagnostic. Replace this file and regenerate the manifest if additional pending add-back model-run seeds are merged. |

After replacing any result CSV or `docs/thesis.pdf`, regenerate the SHA256 manifest:

```bash
python supplementary/reproducibility/make_artifact_manifest.py
```

## Results

The model script writes these two result files:

```text
results/recbole_results_by_seed_<RUN_TAG>.csv
results/recbole_results_summary_<RUN_TAG>.csv
```

`recbole_results_by_seed_*` contains one row per variant, model, and seed. `recbole_results_summary_*` groups by variant and model, then stores the mean and standard deviation for numeric metrics.

The main metric is `NDCG@10`. The scripts also produce `Precision@10`, `Recall@10`, and `MRR@10`. Beyond-accuracy metrics are only added when their switches are enabled.

## Updating after additional add-back runs

When the remaining add-back model runs finish, do the following before final submission:

1. Merge the new per-seed outputs into the aggregate `addback_results.csv`.
2. Recompute any thesis table/figure that reads from `addback_results.csv`, especially Appendix A.10.
3. Rebuild the PDF and replace `docs/thesis.pdf`.
4. Regenerate `supplementary/reproducibility/artifact_manifest_sha256.txt`.
5. Commit the changed CSV, PDF, README if needed, and manifest together.

Keep variant construction and model-run seeds separate in the documentation: `ADDBACK_SEEDS` controls which add-back variant files are generated, while `SEEDS` controls RecBole model-run repetitions.

## Notes for comparing runs

- Keep `NEGATIVE_CANDIDATE_SAMPLE` fixed when comparing experiments.
- Keep the same `RUN_ID` between preparation and model execution.
- Use `RUN_VARIANT_GLOB` for partial reruns instead of editing the source code.
- Use `RUN_MODEL_LIST=BPR` for quick checks before running all models.
- Set `REUSE_ARTIFACTS=0` when cached metrics should not be reused.
- Save recommendations with `SAVE_RECS=1` before using `EVAL_ONLY=1`.
- Sampled-candidate metrics should not be reported as full-catalog ranking metrics.
- The original regimes and size-matched k-core controls use variant-specific test tasks.
- The fixed-test path is intended for the add-back family, not for the size-matched k-core controls.
