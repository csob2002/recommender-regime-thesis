
import gc
import json
import os
import re
import fnmatch
import glob as _glob
import time as _time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ARTIFACTS_DIR,
    BINARIZE_FROM_EXPLICIT,
    Config,
    DATASET_NAME,
    EXPORT_DIR,
    LOG_SPARSITY_PER_MODEL,
    NEGATIVE_CANDIDATE_SAMPLE,
    RECBOLE_DEVICE,
    RECBOLE_EPOCHS,
    RECBOLE_EVAL_BATCH_SIZE,
    RECBOLE_EVAL_BATCH_SIZE_MULTIVAE,
    RECBOLE_TRAIN_BATCH_SIZE,
    RECBOLE_TRAIN_BATCH_SIZE_MULTIVAE,
    RESULTS_DIR,
    RUN_ID,
    RUN_RECBOLE_MODELS,
    SAVE_TOPK_JSON,
    SEED,
    SEEDS,
    TOPK,
    VARIANTS_DIR,
    WINDOWS_DIR,
    common_split_paths,
    create_dataset,
    data_preparation,
    dense_csv,
    df_to_surprise_train_test,
    evaluate_ranking,
    evaluate_ranking_from_saved_topk,
    filter_by_activity_iterative,
    get_model,
    get_trainer,
    init_seed,
    load_variant_split,
    log_sparsity_df,
    maybe_binarize,
    normalize_cols_any,
    raw_aligned_csv,
    raw_aligned_fixitems_csv,
    raw_full_csv,
    recbole_dataset_to_df,
    save_common_split_for_variant_csv,
    sparse_csv,
    user_dense_csv,
    user_sparse_csv,
    _load_surprise,
    _log_pos_stats,
    _write_json_no_overwrite,
    _write_csv_no_overwrite,
)

RECBOLE_EVAL_STEP = int(os.getenv("RECBOLE_EVAL_STEP", "5"))
RECBOLE_EVAL_SAMPLE_NUM = int(os.getenv("RECBOLE_EVAL_SAMPLE_NUM", "200"))
SKIP_FIT_VALIDATION = os.getenv("SKIP_FIT_VALIDATION", "1").strip().lower() in ("1", "true", "yes", "y")
ONLY_WINDOWS = os.getenv("ONLY_WINDOWS", "0").lower() in ("1", "true", "yes")
RUN_TAG = os.getenv("RUN_TAG", "").strip()
OUT_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""
RUN_WINDOWS = os.getenv("RUN_WINDOWS", "0").strip().lower() in ("1", "true", "yes", "y")
REUSE_ARTIFACTS = os.getenv("REUSE_ARTIFACTS", "1").strip().lower() in ("1", "true", "yes", "y")
SAVE_RECS = os.getenv("SAVE_RECS", "1").strip().lower() in ("1", "true", "yes", "y")
EVAL_ONLY = os.getenv("EVAL_ONLY", "0").strip().lower() in ("1", "true", "yes", "y")
OVERWRITE_METRICS = os.getenv("OVERWRITE_METRICS", "1").strip().lower() in ("1", "true", "yes", "y")

COMMON_SPLIT_SHUFFLE = os.getenv("COMMON_SPLIT_SHUFFLE", "0").strip().lower() in ("1", "true", "yes", "y")

RECBOLE_EVAL_STEP = int(os.getenv("RECBOLE_EVAL_STEP", "5"))
RECBOLE_EVAL_SAMPLE_NUM = int(os.getenv("RECBOLE_EVAL_SAMPLE_NUM", "200"))

DEFAULT_MODEL_LIST = ["BPR", "ItemKNN", "LightGCN", "NeuMF", "MultiVAE"]


def _parse_run_model_list():
    # Primary variable used by the pipeline.  RECBOLE_MODELS is kept as a
    # backward-compatible alias because older README snippets used that name.
    raw = os.getenv("RUN_MODEL_LIST", "").strip()
    if not raw:
        raw = os.getenv("RECBOLE_MODELS", "").strip()
    if not raw:
        return list(DEFAULT_MODEL_LIST)
    lookup = {m.lower(): m for m in DEFAULT_MODEL_LIST}
    selected = []
    for part in raw.replace(";", ",").split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in lookup:
            print(f"[WARN] Unknown model in RUN_MODEL_LIST/RECBOLE_MODELS: {part!r}; skipping it")
            continue
        selected.append(lookup[key])
    return selected if selected else list(DEFAULT_MODEL_LIST)


class RBAdapter:
    def __init__(self, model, dataset, device=None, config=None):
        import numpy as np
        import torch
        from recbole.data.interaction import Interaction

        self.model = model
        self.dataset = dataset
        self.torch = torch
        self.Interaction = Interaction

        
        model_param = next(model.parameters(), None)
        if model_param is not None:
            model_dev = model_param.device
        else:
            fallback = str(config["device"]) if (config is not None and "device" in config) else "cpu"
            model_dev = torch.device(fallback)

        self.device = model_dev if device is None else torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

        all_uids = np.arange(dataset.user_num)
        all_iids = np.arange(dataset.item_num)

        uid_tokens = dataset.id2token(self.uid_field, all_uids)
        iid_tokens = dataset.id2token(self.iid_field, all_iids)

        self.raw_to_internal_uid = {str(tok): int(uid) for uid, tok in zip(all_uids, uid_tokens)}
        self.raw_to_internal_iid = {str(tok): int(iid) for iid, tok in zip(all_iids, iid_tokens)}

        default_bs = int(os.getenv("RB_EVAL_BATCH_SIZE", "2048"))
        self.eval_batch_size = int(config["eval_batch_size"]) if (config is not None and "eval_batch_size" in config) else default_bs

        
        self.debug = os.getenv("RB_ADAPTER_DEBUG", "0").strip().lower() in ("1", "true", "yes", "y")
        self.debug_max = int(os.getenv("RB_ADAPTER_DEBUG_MAX", "3"))
        self._debug_printed = 0

    def score_candidates(self, raw_uid, raw_iids):
        import numpy as np
        t = self.torch

        raw_uid = str(raw_uid)
        uid_internal = self.raw_to_internal_uid.get(raw_uid)
        if uid_internal is None:
            return np.zeros(len(raw_iids), dtype=float)

        do_dbg = self.debug and (self._debug_printed < self.debug_max)

        if do_dbg:
            print("[DBG] raw_uid:", raw_uid, "uid_internal:", uid_internal)
            print("[DBG] adapter.device:", self.device)
            mp = next(self.model.parameters(), None)
            print("[DBG] model.device:", (mp.device if mp is not None else self.device))

        internal = []
        valid_pos = []
        for idx, rid in enumerate(raw_iids):
            iid_internal = self.raw_to_internal_iid.get(str(rid), -1)
            internal.append(iid_internal)
            if iid_internal >= 0:
                valid_pos.append(idx)

        out = np.zeros(len(raw_iids), dtype=float)
        if not valid_pos:
            return out

        bs = self.eval_batch_size
        valid_iids = np.array([internal[i] for i in valid_pos], dtype=np.int64)

        with t.no_grad():
            start = 0
            while start < len(valid_iids):
                batch_iids = valid_iids[start:start + bs]

                u = t.tensor([uid_internal] * len(batch_iids), dtype=t.long, device=self.device)
                i = t.tensor(batch_iids, dtype=t.long, device=self.device)

                inter = self.Interaction({self.uid_field: u, self.iid_field: i}).to(self.device)

                if do_dbg and start == 0:
                    print("[DBG] u.device:", u.device, "i.device:", i.device)
                    self._debug_printed += 1

                scores = self.model.predict(inter).view(-1).detach().cpu().numpy()
                for j, s in enumerate(scores):
                    out[valid_pos[start + j]] = float(s)

                start += bs

        return out

    def predict(self, raw_uid, raw_iid, verbose=False):
        class Pred:
            pass
        s = self.score_candidates(raw_uid, [raw_iid])[0]
        p = Pred()
        p.est = float(s)
        return p


def ensure_recbole():
    try:
        import recbole  
        return True
    except Exception:
        print("RecBole is not installed. Install with: pip install recbole")
        return False


def _apply_fixed_implicit_threshold(df: pd.DataFrame, thr=None):


    df = normalize_cols_any(df)
    if len(df) == 0:
        return pd.DataFrame(columns=["user_id", "item_id", "rating"])

    rmin = float(df["rating"].min())
    rmax = float(df["rating"].max())
    if rmin >= -1e-9 and rmax <= 1.0 + 1e-9:
        out = df[["user_id", "item_id"]].drop_duplicates().copy()
        out["rating"] = 1.0
        return out.reset_index(drop=True)

    if thr is None:
        out = df[["user_id", "item_id"]].drop_duplicates().copy()
        out["rating"] = 1.0
        return out.reset_index(drop=True)

    out = df.loc[df["rating"] >= float(thr), ["user_id", "item_id"]].drop_duplicates().copy()
    out["rating"] = 1.0
    return out.reset_index(drop=True)


def _write_inter_df(df: pd.DataFrame, inter_path: Path):
    inter_path.parent.mkdir(parents=True, exist_ok=True)
    df_typed = normalize_cols_any(df).copy()
    df_typed["user_id"] = df_typed["user_id"].astype(str)
    df_typed["item_id"] = df_typed["item_id"].astype(str)
    df_typed["rating"] = 1.0
    df_typed = df_typed.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "rating": "rating:float",
        }
    )
    df_typed.to_csv(inter_path, index=False)


def write_recbole_interactions(csv_path: str, name: str, out_root: Path, seed: int = SEED):


    split_paths = common_split_paths(name)
    if (not split_paths["train_csv"].exists()) or (not split_paths["test_csv"].exists()):
        save_common_split_for_variant_csv(csv_path=str(csv_path), seed=seed, shuffle=COMMON_SPLIT_SHUFFLE, force=False)

    train_df, test_df, split_meta = load_variant_split(name, create_if_missing=False, seed=seed, shuffle=COMMON_SPLIT_SHUFFLE)
    train_df = normalize_cols_any(train_df)
    test_df = normalize_cols_any(test_df)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    if len(full_df) == 0:
        return None, None, train_df, test_df, 0, None, split_meta

    full_bin_preview, used_thr = maybe_binarize(
        full_df,
        enabled=BINARIZE_FROM_EXPLICIT,
        ensure_min_pos_user=0,
    )
    _log_pos_stats(full_bin_preview, f"RecBole common split bin preview ({name}, thr={used_thr})")

    if used_thr in (None, "prebinarized"):
        thr_value = None
    else:
        try:
            thr_value = float(used_thr)
        except Exception:
            thr_value = None

    train_bin = _apply_fixed_implicit_threshold(train_df, thr=thr_value)
    test_bin = _apply_fixed_implicit_threshold(test_df, thr=thr_value)

    _log_pos_stats(train_bin, f"RecBole train bin ({name}, thr={used_thr})")
    _log_pos_stats(test_bin, f"RecBole test bin ({name}, thr={used_thr})")

    if ONLY_WINDOWS or str(name).startswith("win_") or str(name).startswith("win"):
        n_items = int(train_bin["item_id"].nunique()) if len(train_bin) else 0
        max_inter_per_user = min(max(int(0.7 * n_items), 1), 50) if n_items > 0 else 1

        if len(train_bin) and max_inter_per_user >= 1:
            rows_before = len(train_bin)
            capped = []
            for uid, g in train_bin.groupby("user_id", sort=False):
                if len(g) <= max_inter_per_user:
                    capped.append(g)
                else:
                    capped.append(g.sample(n=max_inter_per_user, random_state=int(SEED)))

            train_bin = (
                pd.concat(capped, ignore_index=True)
                .drop_duplicates(subset=["user_id", "item_id"])
                .reset_index(drop=True)
            )
            print(
                f"[WINDOW CAP] {name}: items={n_items}, cap={max_inter_per_user}, "
                f"rows {rows_before:,} -> {len(train_bin):,}"
            )

        rows_before_deg = len(train_bin)
        train_bin = filter_by_activity_iterative(train_bin, min_user=2, min_item=2)
        print(
            f"[WINDOW DEG] {name}: rows {rows_before_deg:,} -> {len(train_bin):,} "
            f"(min_user=2, min_item=2)"
        )
        _log_pos_stats(train_bin, f"RecBole win deg-filter ({name})")

    if len(train_bin) == 0:
        print(f"[WARN] {name}: train split is empty after binarization/filtering (thr={used_thr}).")

    data_path = out_root / f"recbole_{name}"
    dataset_name = re.sub(r"[^0-9A-Za-z_]+", "_", f"{DATASET_NAME}_{name}")
    dataset_dir = data_path / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_inter_path = dataset_dir / f"{dataset_name}.train.inter"
    valid_inter_path = dataset_dir / f"{dataset_name}.valid.inter"
    test_inter_path = dataset_dir / f"{dataset_name}.test.inter"

    empty_inter = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    _write_inter_df(train_bin, train_inter_path)
    _write_inter_df(empty_inter, valid_inter_path)
    _write_inter_df(empty_inter, test_inter_path)

    return dataset_name, data_path, train_bin, test_bin, int(len(train_bin)), used_thr, split_meta


def save_recbole_yaml(dataset_name: str, ds_dir: Path, seed: int) -> str:
    yaml_text = f"""
field_separator: ","
data_path: "{str(ds_dir)}"
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
save_path: "{EXPORT_DIR}/recbole_saved/seed_{seed}"
seed: {seed}
show_progress: False
state: INFO
log_wandb: False
benchmark_filename: [train, valid, test]
# training/eval
epochs: {RECBOLE_EPOCHS}
train_batch_size: {RECBOLE_TRAIN_BATCH_SIZE}
eval_batch_size: {RECBOLE_EVAL_BATCH_SIZE}
eval_step: {RECBOLE_EVAL_STEP}
eval_neg_sample_args:
  distribution: uniform
  sample_num: {RECBOLE_EVAL_SAMPLE_NUM}
eval_args:
  group_by: user
  order: RO
  mode: uni200
metrics: [Recall, NDCG, MRR, Hit]
topk: [{TOPK}]
valid_metric: NDCG@{TOPK}
device: "{RECBOLE_DEVICE}"
""".strip("\n")
    out_path = Path(ds_dir) / f"{dataset_name}_seed{seed}.yaml"
    with open(out_path, "w") as f:
        f.write(yaml_text)
    return str(out_path)


def run_recbole_model_for_variant(csv_path: str, name: str, model_name: str, seed: int = SEED):
    if not ensure_recbole():
        return pd.DataFrame([{
            "variant": name, "model": model_name, "seed": seed,
            f"Precision@{TOPK}": np.nan, f"Recall@{TOPK}": np.nan,
            f"NDCG@{TOPK}": np.nan, f"MRR@{TOPK}": np.nan,
            f"Coverage@{TOPK}": np.nan, "CatalogItems": np.nan,
            f"UniqueRecItems@{TOPK}": np.nan, f"UserCoverage@{TOPK}": np.nan,
            f"EPC@{TOPK}": np.nan, f"EEL@{TOPK}": np.nan,
            "ILS": np.nan, "Diversity": np.nan,
            "RecGini": np.nan, "RecEntropy": np.nan, "rmse": np.nan, "sparsity": np.nan
        }])

    
    art_root = ARTIFACTS_DIR / "recbole" / model_name / f"seed_{seed}" / name
    splits_dir = art_root / "splits"
    recs_dir = art_root / "recs"
    meta_dir = art_root / "meta"
    for _p in [splits_dir, recs_dir, meta_dir]:
        _p.mkdir(parents=True, exist_ok=True)
    metrics_path = art_root / "metrics.json"
    model_state_path = art_root / "model_state.pt"

    
    if EVAL_ONLY:
        import shutil

        train_csv = splits_dir / "train.csv"
        test_csv = splits_dir / "test.csv"

        if (not train_csv.exists()) or (not test_csv.exists()):
            print(f"[EVAL_ONLY][WARN] Missing split files: {train_csv} / {test_csv}. Run training first with SAVE_RECS=1.")
            return pd.DataFrame([{
                "variant": name, "model": model_name, "seed": int(seed), "status": "missing_splits"
            }])

        
        cand = list(recs_dir.glob(f"top{TOPK}_*_{name}.json"))
        if not cand:
            cand = [x for x in recs_dir.glob(f"top{TOPK}_*.json") if name in x.name]
        if not cand:
            cand = list(recs_dir.glob(f"topk_{name}_*.json"))
        if not cand:
            print(f"[EVAL_ONLY][WARN] No Top-K JSON found in {recs_dir}. SAVE_RECS=1 is required during the first run.")
            return pd.DataFrame([{
                "variant": name, "model": model_name, "seed": int(seed), "status": "missing_topk"
            }])

        topk_path = max(cand, key=lambda x: x.stat().st_mtime)
        user_topk = json.loads(topk_path.read_text(encoding="utf-8"))

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        _load_surprise()
        trainset_s, testset_s = df_to_surprise_train_test(train_df, test_df)

        metrics = evaluate_ranking_from_saved_topk(
            user_topk=user_topk,
            trainset=trainset_s,
            testset=testset_s,
            topk=TOPK,
            model_name=f"{model_name}(RB)",
            variant_name=name,
            seed=seed,
        )
        metrics["status"] = "eval_only"
        metrics["topk_file"] = topk_path.name
        metrics["updated_at_unix"] = int(_time.time())

        
        if metrics_path.exists():
            backup = metrics_path.with_name(f"metrics_backup_{int(_time.time())}.json")
            try:
                shutil.copy2(metrics_path, backup)
            except Exception as e:
                print(f"[EVAL_ONLY][WARN] Backup failed: {repr(e)}")

        if OVERWRITE_METRICS or (not metrics_path.exists()):
            metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
        else:
            alt = metrics_path.with_name(f"metrics_recomputed_{int(_time.time())}.json")
            alt.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")

        return pd.DataFrame([metrics])

    if REUSE_ARTIFACTS and metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached["variant"] = name
            cached["model"] = model_name
            cached["seed"] = seed
            return pd.DataFrame([cached])
        except Exception as e:
            print(f"[WARN] Could not load cached metrics: {repr(e)}")

    
    prep = write_recbole_interactions(
        csv_path, name, (ARTIFACTS_DIR / "recbole" / "_data"), seed=seed
    )
    if prep is None:
        return pd.DataFrame([{
            "variant": name,
            "model": model_name,
            "rmse": np.nan,
            "seed": seed,
            "sparsity": np.nan,
            **{k: np.nan for k in [
                f"Precision@{TOPK}", f"Recall@{TOPK}", f"NDCG@{TOPK}", f"MRR@{TOPK}",
                f"Coverage@{TOPK}", "CatalogItems", f"UniqueRecItems@{TOPK}", f"UserCoverage@{TOPK}",
                f"EPC@{TOPK}", f"EEL@{TOPK}", "ILS", "Diversity", "RecGini", "RecEntropy",
                f"1-Gini@{TOPK}", f"ARP@{TOPK}", f"APLT@{TOPK}", f"LTShare@{TOPK}", f"EFD@{TOPK}"
            ]}
        }])

    dataset_name, ds_dir, tr_df_rb, te_df_rb, n_rows_inter, used_thr, split_meta = prep

    if n_rows_inter <= 0:
        print(f"[WARN] variant={name}: empty after binarize (thr={used_thr}). Skipping RecBole training/eval for model={model_name}.")
        return pd.DataFrame([{
            "variant": name,
            "model": model_name,
            "rmse": np.nan,
            "seed": seed,
            "sparsity": np.nan,
            **{k: np.nan for k in [
                f"Precision@{TOPK}", f"Recall@{TOPK}", f"NDCG@{TOPK}", f"MRR@{TOPK}",
                f"Coverage@{TOPK}", "CatalogItems", f"UniqueRecItems@{TOPK}", f"UserCoverage@{TOPK}",
                f"EPC@{TOPK}", f"EEL@{TOPK}", "ILS", "Diversity", "RecGini", "RecEntropy",
                f"1-Gini@{TOPK}", f"ARP@{TOPK}", f"APLT@{TOPK}", f"LTShare@{TOPK}", f"EFD@{TOPK}"
            ]}
        }])

    import torch
    device = RECBOLE_DEVICE
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using cpu")
        device = "cpu"

    
    if model_name.lower() in ("itemknn", "userknn", "knn"):
        device = "cpu"

    
    if str(DATASET_NAME).lower() == "steam" and str(name) == "raw_full" and model_name.lower() == "multivae":
        device = torch.device("cpu")
        print(f"[FORCE-CPU] model={model_name} variant={name} seed={seed} device={device}")

    print(f"[DEVICE] model={model_name} variant={name} seed={seed} device={device}")
    use_gpu = isinstance(device, str) and device.startswith("cuda")
    gpu_id = 0

    cfg_dict = {
    "data_path": str(ds_dir),
    "USER_ID_FIELD": "user_id",
    "ITEM_ID_FIELD": "item_id",
    "field_separator": ",",
    "load_col": {"inter": ["user_id", "item_id", "rating"]},
    "benchmark_filename": ["train", "valid", "test"],
    "epochs": RECBOLE_EPOCHS,
    "train_batch_size": int(RECBOLE_TRAIN_BATCH_SIZE),
    "eval_batch_size": int(RECBOLE_EVAL_BATCH_SIZE),
    "eval_step": RECBOLE_EVAL_STEP,
    "eval_neg_sample_args": {
        "distribution": "uniform",
        "sample_num": RECBOLE_EVAL_SAMPLE_NUM,
    },
    "eval_args": {
        "group_by": "user",
        "order": "RO",
        "mode": "uni200",
    },
    "metrics": ["Recall", "NDCG", "MRR", "Hit"],
    "topk": [TOPK],
    "valid_metric": f"NDCG@{TOPK}",
    "seed": seed,
    "device": device,
    "show_progress": False,
    "save_dataset": False,
    "state": "INFO",
    "log_wandb": False,
    "enable_amp": False,
    }

    if model_name.lower() == "multivae":
        # PDF-aligned defaults: LFM/LastFM uses 512/512, the other main
        # datasets use 1024/1024.  Environment variables still override this.
        ds_lower = str(DATASET_NAME).lower()
        is_lfm = ("lastfm" in ds_lower) or ("lfm" in ds_lower)
        default_mv_train, default_mv_eval = (512, 512) if is_lfm else (1024, 1024)
        mv_train = os.getenv("RECBOLE_TRAIN_BATCH_SIZE_MULTIVAE")
        mv_eval = os.getenv("RECBOLE_EVAL_BATCH_SIZE_MULTIVAE")
        cfg_dict["train_batch_size"] = int(mv_train) if mv_train else int(default_mv_train)
        cfg_dict["eval_batch_size"] = int(mv_eval) if mv_eval else int(default_mv_eval)
        cfg_dict["enable_amp"] = False
        cfg_dict["mlp_hidden_size"] = [16]
        cfg_dict["latent_dimension"] = 8

        
        if str(DATASET_NAME).lower() == "steam" and str(name) == "raw_full":
            cfg_dict["device"] = "cpu"
            device = "cpu"

    if SKIP_FIT_VALIDATION:
        cfg_dict["eval_step"] = 10**9

    if str(model_name).lower() == "lightgcn":
        cfg_dict["enable_amp"] = False
        cfg_dict["embedding_size"] = 16
        cfg_dict["n_layers"] = 1

    if str(model_name).lower() == "bpr":
        cfg_dict["embedding_size"] = 16

    if str(model_name).lower() == "neumf":
        cfg_dict["mf_embedding_size"] = 16
        cfg_dict["mlp_embedding_size"] = 16
        cfg_dict["mlp_hidden_size"] = [32, 16]

    if str(model_name).lower() == "itemknn":
        cfg_dict["k"] = 20
        cfg_dict["shrink"] = 0.0
        cfg_dict["knn_method"] = "item"

    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_dict=cfg_dict,
    )
    if str(DATASET_NAME).lower() == "steam" and str(name) == "raw_full" and model_name.lower() == "multivae":
        cpu_device = torch.device("cpu")
        config["device"] = cpu_device
        device = torch.device("cpu")

    print(f"[FINAL-DEVICE] dataset={DATASET_NAME} variant={name} model={model_name} device={device} cfg_device={config['device']}")

    
    init_seed(config["seed"], reproducibility=True)
    rb_dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, rb_dataset)

    def _dl_ds(dl):
        if dl is None:
            return None
        return getattr(dl, "_dataset", getattr(dl, "dataset", None))

    train_ds = _dl_ds(train_data)
    valid_ds = _dl_ds(valid_data)
    test_ds = _dl_ds(test_data)

    va_df_rb = recbole_dataset_to_df(valid_ds) if valid_ds is not None else pd.DataFrame(columns=["user_id", "item_id", "rating"])
    te_internal_df_rb = recbole_dataset_to_df(test_ds) if test_ds is not None else pd.DataFrame(columns=["user_id", "item_id", "rating"])

    print("len train split (common external):", len(tr_df_rb))
    print("len valid split (internal benchmark):", len(va_df_rb))
    print("len test  split (internal benchmark):", len(te_internal_df_rb))

    rb_train_pairs = set(zip(tr_df_rb["user_id"], tr_df_rb["item_id"]))
    rb_test_pairs = set(zip(te_df_rb["user_id"], te_df_rb["item_id"]))
    print(f"[SPLIT-CHECK][{name}][{model_name}][seed={seed}] common train vs common test overlap = {len(rb_train_pairs & rb_test_pairs)}")
    if len(rb_train_pairs & rb_test_pairs) > 0:
        raise RuntimeError(
            "Invalid split: common train and test overlap on user-item pairs."
        )

    
    try:
        tr_path = splits_dir / "train.csv"
        va_path = splits_dir / "valid.csv"
        te_path = splits_dir / "test.csv"
        if not tr_path.exists():
            tr_df_rb.to_csv(tr_path, index=False)
        if not va_path.exists():
            va_df_rb.to_csv(va_path, index=False)
        if not te_path.exists():
            te_df_rb.to_csv(te_path, index=False)
    except Exception as e:
        print(f"[WARN] Could not save split files: {repr(e)}")

    
    try:
        meta_path = meta_dir / "meta.json"
        if not meta_path.exists():
            _write_json_no_overwrite(
                {
                    "dataset_name": DATASET_NAME,
                    "run_id": RUN_ID,
                    "variant": name,
                    "model": model_name,
                    "seed": seed,
                    "used_threshold": used_thr,
                    "n_interactions_binarized_train": int(n_rows_inter),
                    "n_interactions_test": int(len(te_df_rb)),
                    "split_meta": split_meta,
                },
                meta_path,
                indent=2,
            )
    except Exception:
        pass
    

    
    
    model_class = get_model(config["model"])
    model = model_class(config, train_data.dataset).to(device)
    print("RecBole model class for", model_name, ":", model.__class__.__name__)

    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model)

    

    fit_valid_data = None if (SKIP_FIT_VALIDATION or va_df_rb.empty) else valid_data
    print("[FIT] skip internal validation:", SKIP_FIT_VALIDATION)
    print("[TIME] before trainer.fit", _time.strftime("%Y-%m-%d %H:%M:%S"))
    trainer.fit(train_data, fit_valid_data)
    print("[TIME] after trainer.fit", _time.strftime("%Y-%m-%d %H:%M:%S"))

    
    try:
        if not model_state_path.exists():
            torch.save(model.state_dict(), model_state_path)
    except Exception as e:
        print(f"[WARN] Could not save model state: {repr(e)}")

    
    full_df_rb = pd.concat([tr_df_rb, te_df_rb], ignore_index=True)
    n_users = full_df_rb["user_id"].nunique()
    n_items = full_df_rb["item_id"].nunique()
    sparsity_val = 1.0 - (len(full_df_rb) / (n_users * n_items) if (n_users and n_items) else 0.0)

    if LOG_SPARSITY_PER_MODEL:
        log_sparsity_df(tr_df_rb, f"recbole:{name}/train(common)")

    _load_surprise()

    
    trainset_s, testset_s = df_to_surprise_train_test(tr_df_rb, te_df_rb)

    adapter = RBAdapter(model=model, dataset=train_data.dataset, device=None, config=config)

    mp = next(model.parameters(), None)
    print("model device:", (mp.device if mp is not None else adapter.device))
    print("adapter device:", adapter.device)

    
    import traceback
    try:
        if te_df_rb.empty:
            print(f"[DEBUG] variant={name}, model={model_name}: te_df_rb is empty; no debug sample available.")
        else:
            debug_uid = te_df_rb["user_id"].iloc[0]
            debug_items = te_df_rb["item_id"].unique()[:10]

            print(f"\n[DEBUG] variant={name}, model={model_name}, debug_uid={debug_uid}")
            print("  internal uid:", adapter.raw_to_internal_uid.get(str(debug_uid), None))

            for iid in debug_items:
                internal_iid = adapter.raw_to_internal_iid.get(str(iid), None)
                print("  item", iid, "-> internal", internal_iid)

            for iid in debug_items:
                s = adapter.predict(debug_uid, iid).est
                print(f"  item {iid}: score={s:.6f}")
    except Exception as e:
        print(f"[DEBUG] scoring check failed for {name}, {model_name}: {repr(e)}")
        traceback.print_exc()

    print("[TIME] before evaluate", _time.strftime("%Y-%m-%d %H:%M:%S"))
    out_metrics = evaluate_ranking(
        adapter,
        trainset=trainset_s,
        testset=testset_s,
        topk=TOPK,
        save_ranked=(SAVE_TOPK_JSON or SAVE_RECS),
        model_name=f"{model_name}(RB)",
        variant_name=name,
        save_path=str(recs_dir),
        neg_sample=NEGATIVE_CANDIDATE_SAMPLE,
        seed=seed,
    )
    print("[TIME] after evaluate", _time.strftime("%Y-%m-%d %H:%M:%S"))

    row = {
        "variant": name,
        "model": model_name,
        "rmse": np.nan,
        "seed": seed,
        "sparsity": sparsity_val,
    }
    row.update(out_metrics)

    ndcg_key = f"NDCG@{TOPK}"
    rec_key = f"Recall@{TOPK}"
    prec_key = f"Precision@{TOPK}"

    print(
        f"[RecBole][variant={name}][model={model_name}][seed={seed}] "
        f"{ndcg_key}={row.get(ndcg_key, float('nan')):.8f} "
        f"{rec_key}={row.get(rec_key, float('nan')):.8f} "
        f"{prec_key}={row.get(prec_key, float('nan')):.8f}"
    )
    print(
        f"   Coverage@{TOPK}={row.get(f'Coverage@{TOPK}', float('nan')):.4f} "
        f"UserCoverage@{TOPK}={row.get(f'UserCoverage@{TOPK}', float('nan')):.4f} "
        f"EPC@{TOPK}={row.get(f'EPC@{TOPK}', float('nan')):.4f} "
        f"EEL@{TOPK}={row.get(f'EEL@{TOPK}', float('nan')):.4f}"
    )

    
    try:
        if OVERWRITE_METRICS or (not metrics_path.exists()):
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(row, indent=2, ensure_ascii=True),
                encoding="utf-8"
            )
        else:
            alt = metrics_path.with_name(f"metrics_recomputed_{int(_time.time())}.json")
            alt.write_text(
                json.dumps(row, indent=2, ensure_ascii=True),
                encoding="utf-8"
            )
    except Exception as e:
        print(f"[WARN] Could not save metrics.json: {repr(e)}")

    return pd.DataFrame([row])


def _apply_variant_glob_filter(variants, glob_raw: str):
    glob_raw = (glob_raw or "").strip()
    if not glob_raw:
        return list(variants)

    patterns = [p.strip() for p in glob_raw.replace(";", ",").split(",") if p.strip()]
    if not patterns:
        return list(variants)

    out = []
    for pth, nm in variants:
        fname = Path(pth).name
        if any(fnmatch.fnmatch(fname, patt) or fnmatch.fnmatch(nm, patt) for patt in patterns):
            out.append((pth, nm))
    return out


def _env_bool_value(key: str, default: bool = False) -> bool:
    v = os.getenv(key, None)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def _variant_paths_from_globs(glob_raw: str):
    """Resolve RUN_VARIANT_GLOB directly against VARIANTS_DIR.

    Patterns can match file names such as ratings_td*.csv, variant names such as
    td60, or absolute paths. Multiple patterns can be separated with commas or
    semicolons.
    """
    patterns = [p.strip() for p in (glob_raw or "").replace(";", ",").split(",") if p.strip()]
    out = []
    seen = set()
    all_variant_files = sorted(VARIANTS_DIR.glob("ratings_*.csv"))

    for patt in patterns:
        matches = []
        patt_path = Path(patt).expanduser()
        if patt_path.is_absolute() or ("/" in patt) or ("\\" in patt):
            matches = [Path(x) for x in sorted(_glob.glob(str(patt_path)))]
        else:
            matches = sorted(VARIANTS_DIR.glob(patt))
            if not matches:
                for p in all_variant_files:
                    nm = p.stem.replace("ratings_", "")
                    if fnmatch.fnmatch(p.name, patt) or fnmatch.fnmatch(nm, patt):
                        matches.append(p)

        for p in matches:
            if not p.exists() or p.suffix.lower() != ".csv":
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append((str(p), p.stem.replace("ratings_", "")))
    return out


def _dedupe_variants(variants):
    out = []
    seen = set()
    for pth, nm in variants:
        key = (str(Path(pth).resolve()), str(nm))
        if key in seen:
            continue
        seen.add(key)
        out.append((pth, nm))
    return out


if RUN_RECBOLE_MODELS:
    model_list = _parse_run_model_list()
    print("\nRecBole models (implicit): " + ", ".join(model_list))
    per_seed_rb = []

    run_variant_glob = os.getenv("RUN_VARIANT_GLOB", "").strip()

    run_only_addback_variants = _env_bool_value("RUN_ONLY_ADDBACK_VARIANTS", False)
    run_addback_variants = _env_bool_value("RUN_ADDBACK_VARIANTS", False)
    addback_source_k = int(os.getenv("ADDBACK_SOURCE_K", "5"))
    addback_target_k = int(os.getenv("ADDBACK_TARGET_K", "50"))

    run_only_kcore_sizematch = _env_bool_value("RUN_ONLY_KCORE_SIZE_MATCHED_VARIANTS", False)
    run_kcore_sizematch = _env_bool_value("RUN_KCORE_SIZE_MATCHED_VARIANTS", False)
    kcore_size_match_target = int(os.getenv("KCORE_SIZE_MATCH_TARGET", "50"))
    run_kcore_size_match_target = _env_bool_value("RUN_KCORE_SIZE_MATCH_TARGET", True)

    run_only_taildrop_stress = _env_bool_value("RUN_ONLY_TAILDROP_STRESS", False)
    run_taildrop_stress = _env_bool_value("RUN_TAILDROP_STRESS_VARIANTS", False)

    only_modes = [
        name for name, enabled in [
            ("RUN_ONLY_ADDBACK_VARIANTS", run_only_addback_variants),
            ("RUN_ONLY_KCORE_SIZE_MATCHED_VARIANTS", run_only_kcore_sizematch),
            ("RUN_ONLY_TAILDROP_STRESS", run_only_taildrop_stress),
        ]
        if enabled
    ]
    if len(only_modes) > 1:
        raise RuntimeError(f"Only one RUN_ONLY_* mode can be enabled at a time: {only_modes}")

    if run_only_kcore_sizematch and not run_variant_glob:
        patterns = [f"ratings_kcore*_sizematch_kcore{kcore_size_match_target}_seed*.csv"]
        if run_kcore_size_match_target:
            patterns.insert(0, f"ratings_kcore{kcore_size_match_target}_plain.csv")
        run_variant_glob = ",".join(patterns)

    if run_only_taildrop_stress and not run_variant_glob:
        run_variant_glob = "ratings_td*.csv"

    if run_only_addback_variants and not run_variant_glob:
        run_variant_glob = (
            f"ratings_kcore{addback_target_k}_plain.csv,"
            f"ratings_kcore{addback_target_k}_addback_from_kcore{addback_source_k}_p*_seed*.csv"
        )

    if run_variant_glob:
        variants = _variant_paths_from_globs(run_variant_glob)
        print(f"[RUN_VARIANT_GLOB] Selected {len(variants)} variant(s) using {run_variant_glob!r}")
        for pth, nm in variants:
            print(f"  - {nm}: {pth}")
    else:
        run_item_variants = _env_bool_value("RUN_ITEM_VARIANTS", True)
        run_user_variants = _env_bool_value("RUN_USER_VARIANTS", True)
        run_raw_full_variant = _env_bool_value("RUN_RAW_FULL_VARIANT", False)

        variants_headtail = []

        if run_raw_full_variant:
            variants_headtail += [(raw_full_csv, "raw_full")]

        if run_item_variants:
            variants_headtail += [
                (raw_aligned_csv, "raw_aligned"),
                (sparse_csv, "sparse"),
                (dense_csv, "dense"),
            ]

        if run_user_variants:
            variants_headtail += [
                (raw_aligned_fixitems_csv, "raw_aligned_fixitems"),
                (user_sparse_csv, "user_sparse"),
                (user_dense_csv, "user_dense"),
            ]

        _ok = []
        for pth, nm in variants_headtail:
            if Path(pth).exists():
                _ok.append((pth, nm))
            else:
                print(f"[WARN] Missing variant file, skipping: {nm} -> {pth}")
        variants_headtail = _ok

        variants_windows = [
            (str(WINDOWS_DIR / "ratings_win_TL.csv"), "win_TL"),
            (str(WINDOWS_DIR / "ratings_win_TR.csv"), "win_TR"),
            (str(WINDOWS_DIR / "ratings_win_BL.csv"), "win_BL"),
            (str(WINDOWS_DIR / "ratings_win_BR.csv"), "win_BR"),
            (str(WINDOWS_DIR / "ratings_win_MID.csv"), "win_MID"),
        ]

        _wok = []
        for pth, nm in variants_windows:
            if Path(pth).exists():
                _wok.append((pth, nm))
            else:
                print(f"[WARN] Missing window file, skipping: {nm} -> {pth}")
        variants_windows = _wok

        run_kcore_variants = _env_bool_value("RUN_KCORE_VARIANTS", True)
        run_kcore_plain_only = _env_bool_value("RUN_KCORE_PLAIN_ONLY", True)

        variants_kcore = []
        if run_kcore_variants:
            if run_kcore_plain_only:
                for p in sorted(VARIANTS_DIR.glob("ratings_kcore*_plain.csv")):
                    nm = p.stem.replace("ratings_", "")
                    variants_kcore.append((str(p), nm))
            else:
                raw_kcore = VARIANTS_DIR / "ratings_raw_kcore_aligned.csv"
                if raw_kcore.exists():
                    variants_kcore.append((str(raw_kcore), "raw_kcore_aligned"))

                for p in sorted(VARIANTS_DIR.glob("ratings_kcore*.csv")):
                    nm = p.stem.replace("ratings_", "")
                    if "addback" in nm and not run_addback_variants:
                        continue
                    if "sizematch" in nm and not run_kcore_sizematch:
                        continue
                    variants_kcore.append((str(p), nm))

        run_randdrop_variants = _env_bool_value("RUN_RANDDROP_VARIANTS", False)
        variants_randdrop = []
        if run_randdrop_variants:
            for p in sorted(VARIANTS_DIR.glob("ratings_randdrop*.csv")):
                nm = p.stem.replace("ratings_", "")
                variants_randdrop.append((str(p), nm))

        variants_addback = []
        if run_addback_variants:
            base_target = VARIANTS_DIR / f"ratings_kcore{addback_target_k}_plain.csv"
            if base_target.exists():
                variants_addback.append((str(base_target), f"kcore{addback_target_k}_plain"))
            for p in sorted(VARIANTS_DIR.glob(f"ratings_kcore{addback_target_k}_addback_from_kcore{addback_source_k}_p*_seed*.csv")):
                nm = p.stem.replace("ratings_", "")
                variants_addback.append((str(p), nm))

        variants_sizematch = []
        if run_kcore_sizematch:
            target = VARIANTS_DIR / f"ratings_kcore{kcore_size_match_target}_plain.csv"
            if run_kcore_size_match_target and target.exists():
                variants_sizematch.append((str(target), f"kcore{kcore_size_match_target}_plain"))
            for p in sorted(VARIANTS_DIR.glob(f"ratings_kcore*_sizematch_kcore{kcore_size_match_target}_seed*.csv")):
                nm = p.stem.replace("ratings_", "")
                variants_sizematch.append((str(p), nm))

        variants_taildrop = []
        if run_taildrop_stress:
            for p in sorted(VARIANTS_DIR.glob("ratings_td*.csv")):
                nm = p.stem.replace("ratings_", "")
                variants_taildrop.append((str(p), nm))

        variants = variants_windows if ONLY_WINDOWS else (
            variants_headtail
            + variants_kcore
            + variants_randdrop
            + variants_addback
            + variants_sizematch
            + variants_taildrop
            + (variants_windows if RUN_WINDOWS else [])
        )
        variants = _dedupe_variants(variants)

    if not variants:
        print("[WARN] No runnable variants were selected.")

    for seed in SEEDS:
        for pth, nm in variants:
            for mdl in model_list:
                print(f"\n[RUN] RecBole model={mdl} variant={nm} seed={seed}")
                r = run_recbole_model_for_variant(pth, nm, mdl, seed=seed)
                if r is not None and not r.empty:
                    per_seed_rb.append(r)
                gc.collect()

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not per_seed_rb:
        print("No RecBole results to aggregate.")
    else:
        recbole_df = pd.concat(per_seed_rb, ignore_index=True)

        out_rb_by_seed = RESULTS_DIR / f"recbole_results_by_seed{OUT_SUFFIX}.csv"
        _write_csv_no_overwrite(recbole_df, out_rb_by_seed)

        
        id_cols = {"variant", "model", "seed"}
        metrics_cols = [
            c for c in recbole_df.columns
            if c not in id_cols and pd.api.types.is_numeric_dtype(recbole_df[c])
        ]

        if not metrics_cols:
            print("No numeric metric columns found, skipping summary.")
            print(f"RecBole per-seed saved to {out_rb_by_seed}")
        else:
            agg_mean = recbole_df.groupby(["variant", "model"], as_index=False)[metrics_cols].mean()
            agg_std = recbole_df.groupby(["variant", "model"], as_index=False)[metrics_cols].std()
            agg = agg_mean.merge(agg_std, on=["variant", "model"], suffixes=("_mean", "_sd"))

            out_rb_summary = RESULTS_DIR / f"recbole_results_summary{OUT_SUFFIX}.csv"
            _write_csv_no_overwrite(agg, out_rb_summary)

            print(f"RecBole per-seed saved to {out_rb_by_seed}")
            print(f"RecBole summary saved to {out_rb_summary}")
