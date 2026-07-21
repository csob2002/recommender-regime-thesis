# Recommender Regime Pipeline

This repository contains the experiment code and cleaned result tables used for the MSc thesis:

**Exploring Performance Variability in Recommender Systems: The Role of Sparsity and Dataset Characteristics**

The pipeline prepares controlled user-item interaction regimes, trains RecBole models, and evaluates Top-K recommendation quality with a shared external evaluator.

## Repository contents

| Path | Purpose |
|---|---|
| `common.py` | Shared configuration, input normalization, splitting, candidate sampling, ranking metrics, and output paths. |
| `prepare_variants.py` | Builds fixed-users, fixed-items, k-core, random-drop, size-matched k-core, tail-drop stress, and add-back variants. |
| `recbole_models.py` | Trains and evaluates BPR, ItemKNN, LightGCN, NeuMF, and MultiVAE. |
| `preprocessing_lastfm.py` | Converts Last.fm play-count data into implicit interactions. |
| `preprocessing_online_retail.py` | Converts Online Retail transactions into implicit interactions. |
| `preprocessing_steam.py` | Converts Steam review records into implicit interactions. |
| `requirements.txt` | Python dependencies used by the recorded experiment environment. |
| `results/main_results.csv` | Main regime-family results. |
| `results/results_kcore_sizematch.csv` | Size-matched k-core control. |
| `results/results_aggressive_taildrop.csv` | Auxiliary fixed-users tail-drop stress test. |
| `results/addback_results.csv` | KC50 add-back diagnostic used in Appendix A.10. |

Raw datasets, generated variants, RecBole run directories, model checkpoints, and saved recommendation lists are not included.

## Experiment overview

The experiments use implicit-feedback Top-K recommendation.

- Models: BPR, ItemKNN, LightGCN, NeuMF, and MultiVAE
- Main metric: NDCG@10
- Additional accuracy metrics: Precision@10, Recall@10, and MRR@10
- Evaluation: one held-out positive and 1000 sampled negative candidates per evaluable user
- Split: user-level leave-one-out after variant construction
- Timestamped datasets: chronological leave-one-out when a usable timestamp is available
- Non-timestamped datasets: seeded random leave-one-out
- Main model-run seeds: 42, 43, and 44
- Training: fixed 20-epoch configurations without a separate tuning sweep

The reported scores are sampled-candidate results, not full-catalog ranking scores.

## Datasets and preprocessing

The pipeline was used with four datasets:

- LFM 2B subset
- MovieLens 20M
- Online Retail
- Steam

The expected interaction format is:

```text
user_id,item_id,rating[,timestamp]
```

`timestamp` is optional.

Explicit ratings can be converted into implicit positives with:

```text
rating >= ceil(0.8 * RATING_MAX)
```

For a 1-5 rating scale, this keeps ratings 4 and 5.

### Last.fm

```bash
python preprocessing_lastfm.py --in inter.txt --out ratings.csv
```

### Online Retail

```bash
python preprocessing_online_retail.py \
  --in OnlineRetail.csv \
  --out ratings_raw_full.csv
```

### Steam

Inspect the source format:

```bash
python preprocessing_steam.py \
  --in steam_reviews.json \
  --out rating.csv \
  --inspect
```

Convert the complete file:

```bash
python preprocessing_steam.py \
  --in steam_reviews.json \
  --out rating.csv \
  --db steam_reviews.sqlite \
  --batch-size 5000
```

## Installation

The recorded environment used Python 3.9, RecBole 1.2.1, and PyTorch 2.6.0.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CPU execution:

```bash
export RECBOLE_DEVICE=cpu
```

## Output layout

Generated runs are written under:

```text
$EXPORT_BASE/$DATASET_NAME/$RUN_ID/
```

The main subdirectories are:

```text
variants/    materialized ratings_*.csv files
splits/      train.csv, test.csv, and split metadata
artifacts/   RecBole inputs, model states, metrics, and optional Top-K files
results/     per-seed and aggregate result CSVs
meta/        run-level metadata
```

Use the same `RUN_ID` for variant preparation and model execution.

The repository-level `results/` directory is different from a run-local `results/` directory. It contains cleaned reporting tables copied from completed experiments.

## Main environment variables

| Variable | Purpose |
|---|---|
| `INPUT_RATINGS_PATH` | Input interaction file. |
| `EXPORT_BASE` | Base directory for generated runs. |
| `DATASET_NAME` | Dataset label used in output paths. |
| `RUN_ID` | Shared identifier for preparation and model execution. |
| `RUN_TAG` | Optional suffix for result filenames. |
| `SEED` | Main construction and split seed. |
| `SEEDS` | Comma-separated model-run seeds, for example `42,43,44`. |
| `TOPK` | Ranking cutoff; the thesis uses `10`. |
| `NEGATIVE_CANDIDATE_SAMPLE` | Number of sampled negative candidates; the thesis uses `1000`. |
| `IMPLICIT_MIN_POS_PER_USER` | Optional minimum number of positives retained per user. |
| `BINARIZE_FROM_EXPLICIT` | Enables explicit-to-implicit conversion. |
| `BINARY_THRESHOLD_FRACTION` | Fraction of `RATING_MAX` used as the positivity threshold. |
| `RECBOLE_DEVICE` | Device used by RecBole, for example `cpu` or `cuda:0`. |
| `RUN_MODEL_LIST` | Comma-separated subset of models to run. |

## Standard workflow

Set the run identity:

```bash
export EXPORT_BASE=/path/to/exports
export DATASET_NAME=my_dataset
export INPUT_RATINGS_PATH=/path/to/ratings.csv
export RUN_ID=my_run_$(date +%Y%m%d_%H%M%S)
export RUN_TAG=my_run

export SEED=42
export SEEDS="42,43,44"
export TOPK=10
export NEGATIVE_CANDIDATE_SAMPLE=1000
```

Prepare variants and splits:

```bash
export RUN_PREPARE_VARIANTS=1
export RUN_RECBOLE_MODELS=0

python -u prepare_variants.py 2>&1 | tee "prepare_${RUN_ID}.log"
```

Train and evaluate models:

```bash
export RUN_PREPARE_VARIANTS=0
export RUN_RECBOLE_MODELS=1

export RUN_MODEL_LIST="BPR,ItemKNN,LightGCN,NeuMF,MultiVAE"
export RECBOLE_DEVICE=cuda:0
export RECBOLE_EPOCHS=20
export SAVE_RECS=1

python -u recbole_models.py 2>&1 | tee "recbole_${RUN_ID}.log"
```

## Variant families

| Family | Main preparation settings |
|---|---|
| Fixed-users item filtering | `PREP_ITEM_VARIANTS=1` |
| Fixed-items user filtering | `PREP_USER_VARIANTS=1` |
| Plain k-core | `PREP_KCORE_VARIANTS=1`, `KCORE_KS="5,10,50"`, `KCORE_SAVE_PLAIN=1` |
| Random drop | `PREP_RANDOM_DROP_VARIANTS=1`, `RANDDROP_PCTS="0.2,0.4,0.6"` |
| Tail-drop stress | `PREP_ONLY_TAILDROP_STRESS=1`, `TAILDROP_STRESS_PCTS="0.4,0.6,0.8"` |
| Size-matched k-core | `PREP_ONLY_KCORE_SIZE_MATCH=1`, `KCORE_SIZE_MATCH_TARGET=50` |
| KC50 add-back | `PREP_ONLY_ADDBACK_VARIANTS=1`, `ADDBACK_SOURCE_K=5`, `ADDBACK_TARGET_K=50` |

Important details:

- Fixed-users variants share an aligned user set while the item catalog changes.
- Fixed-items variants approximately control the item catalog while the retained user population changes.
- KC5, KC10, and KC50 are separate minimum-degree-filtered graphs and are not aligned across `k`.
- Random-drop variants keep the selected user-item support while interactions are removed subject to degree constraints.
- Size matching controls user and item cardinality only. It does not match interaction count, degree distribution, topology, popularity, or exact identities.
- Add-back can restore users and items as well as interactions.

## Add-back experiment

The add-back diagnostic starts from KC50 and reintroduces interactions that are present in KC5 but absent from KC50.

The tested batches are:

```text
10%, 25%, 50%, 100%
```

The main settings are:

```bash
export PREP_ONLY_ADDBACK_VARIANTS=1
export ADDBACK_SOURCE_K=5
export ADDBACK_TARGET_K=50
export ADDBACK_PCTS="0.10,0.25,0.50,1.00"
export ADDBACK_COMMON_TEST_FROM_TARGET=1
export ADDBACK_COMMON_NEGATIVES=1
```

The submitted aggregate uses the following construction-seed coverage:

- LFM 2B subset: seed 42
- MovieLens 20M: seed 42
- Online Retail: seeds 42, 43, and 44
- Steam: seeds 42, 43, and 44

Construction seeds and model-run seeds are different sources of randomness.

The add-back experiment is a diagnostic rather than a causal test. Adding KC5-minus-KC50 interactions can also restore users and items, so the resulting variants do not isolate edge quantity under fixed support. The experiment also lacks matched random, degree-matched, and popularity-matched edge-addition controls.

## Result files

### `results/main_results.csv`

Contains the main results for:

- fixed-users item filtering
- fixed-items user filtering
- plain k-core
- random-drop fixed support

### `results/results_kcore_sizematch.csv`

Contains KC5 and KC10 variants reduced to the user and item cardinality of KC50.

### `results/results_aggressive_taildrop.csv`

Contains the auxiliary fixed-users tail-drop stress levels:

```text
40%, 60%, 80%
```

### `results/addback_results.csv`

Contains the final KC50 add-back rows used in Appendix A.10, including:

- model-level metrics
- construction-seed suffixes where available
- retained user, item, and interaction counts
- density and sparsity
- average user and item degree
- run identifiers

## Interpretation limits

The experiments compare reconstructed benchmark regimes. Several transformations change users, items, interactions, and the evaluation universe at the same time.

In particular:

- higher density does not imply more useful collaborative information;
- k-core variants are not the same task at different densities;
- sampled-candidate scores depend on the retained catalog;
- size matching does not control graph topology or popularity structure;
- add-back does not keep user and item support fixed;
- the fixed 20-epoch setup is intended for consistent regime comparison, not leaderboard claims.
