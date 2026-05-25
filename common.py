
from pathlib import Path
import time
import os
import json
import gc
import math
from collections import defaultdict, Counter
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import joblib

HOME = Path.home()


INPUT_RATINGS_PATH = os.getenv("INPUT_RATINGS_PATH", str(HOME / "datasets/movielens/rating.csv"))
EXPORT_BASE = os.getenv("EXPORT_BASE", str(HOME / "exports_sparsity_new"))

def _infer_dataset_name_from_path(path_str: str) -> str:
    p = (path_str or "").lower()
    if "lastfm" in p or "last-fm" in p or "last_fm" in p:
        return "lastfm"
    if "movielens" in p or "ml-" in p or "ml_" in p:
        return "movielens"
    if "steam" in p:
        return "steam"
    return "movielens"




_DATASET_ENV = (os.getenv("DATASET_NAME", "") or "").strip()
DATASET_NAME = _DATASET_ENV if _DATASET_ENV else _infer_dataset_name_from_path(INPUT_RATINGS_PATH)


def _envflag(name, default):
    v = os.getenv(name)
    return (str(v).strip().lower() in ("1", "true", "yes", "y")) if v is not None else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception:
        print(f"[WARN] Invalid integer env {name}={v!r}; using default {default}")
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        print(f"[WARN] Invalid float env {name}={v!r}; using default {default}")
        return float(default)


def _env_int_list(name: str, default_list):
    raw = os.getenv(name, "").strip()
    if not raw:
        return [int(x) for x in default_list]

    out = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            print(f"[WARN] Invalid seed value in {name}: {part!r}; skipping it")

    return out if out else [int(x) for x in default_list]


RUN_ID = (os.getenv("RUN_ID", time.strftime("%Y%m%d_%H%M%S")) or time.strftime("%Y%m%d_%H%M%S")).strip()
RUN_ROOT = Path(EXPORT_BASE) / DATASET_NAME / RUN_ID

EXPORT_DIR = str(RUN_ROOT)

TOPK = _env_int("TOPK", 10)
TEST_SIZE = _env_float("TEST_SIZE", 0.2)
RATING_SCALE = (
    _env_float("RATING_MIN", 1.0),
    _env_float("RATING_MAX", 5.0),
)



SEED = _env_int("SEED", 42)
SEEDS = _env_int_list("SEEDS", [SEED])


NEGATIVE_CANDIDATE_SAMPLE = int(os.getenv("NEGATIVE_CANDIDATE_SAMPLE", "1000"))


RAW_SAFE = False            



COMPUTE_ILS = os.getenv("COMPUTE_ILS", "0").strip().lower() in ("1", "true", "yes", "y")


COMPUTE_BEYOND_ACCURACY = os.getenv("COMPUTE_BEYOND_ACCURACY", "0").strip().lower() in ("1", "true", "yes", "y")


LOG_SPARSITY_PER_MODEL = os.getenv("LOG_SPARSITY_PER_MODEL", "0").strip().lower() in ("1", "true", "yes", "y")


SURPRISE_ADD_NEGATIVES = True
SURPRISE_NEG_PER_POS = 4          
SURPRISE_NEG_MAX_PER_USER = 500

SURPRISE_IMPLICIT = True                   
IMPLICIT_MIN_POS_PER_USER = _env_int("IMPLICIT_MIN_POS_PER_USER", 0)  
BINARY_THRESHOLD_FRACTION = _env_float("BINARY_THRESHOLD_FRACTION", 0.8)  
BINARIZE_FROM_EXPLICIT = _envflag("BINARIZE_FROM_EXPLICIT", DATASET_NAME.lower() in ("movielens", "ml-1m", "ml-20m"))  


RUN_RECBOLE_MODELS = _envflag("RUN_RECBOLE_MODELS", True)
_RECBOLE_RATING_THRESHOLD_ENV = os.getenv("RECBOLE_RATING_THRESHOLD", "").strip()
RECBOLE_RATING_THRESHOLD = float(_RECBOLE_RATING_THRESHOLD_ENV) if _RECBOLE_RATING_THRESHOLD_ENV else None
RECBOLE_EPOCHS = int(os.getenv("RECBOLE_EPOCHS", "20"))
RECBOLE_DEVICE = os.getenv("RECBOLE_DEVICE", "cpu")
RECBOLE_USE_YAML = True
USE_RECOBOLE_RS_SPLIT = True


RECBOLE_TRAIN_BATCH_SIZE = int(os.getenv("RECBOLE_TRAIN_BATCH_SIZE", "1024"))
RECBOLE_EVAL_BATCH_SIZE = int(os.getenv("RECBOLE_EVAL_BATCH_SIZE", "2048"))
RECBOLE_ENABLE_AMP = (os.getenv("RECBOLE_ENABLE_AMP", "0").strip().lower() in ("1", "true", "yes", "y"))
RECBOLE_TRAIN_BATCH_SIZE_MULTIVAE = int(os.getenv("RECBOLE_TRAIN_BATCH_SIZE_MULTIVAE", "512"))
RECBOLE_EVAL_BATCH_SIZE_MULTIVAE = int(os.getenv("RECBOLE_EVAL_BATCH_SIZE_MULTIVAE", "1024"))

RUN_PREPARE_VARIANTS = _envflag("RUN_PREPARE_VARIANTS", True)
RUN_SPARSITY_REPORTS = _envflag("RUN_SPARSITY_REPORTS", True)
RUN_SURPRISE_BASELINES = _envflag("RUN_SURPRISE_BASELINES", True)
RUN_RELOAD_OR_TRAIN = _envflag("RUN_RELOAD_OR_TRAIN", False)
SAVE_TOPK_JSON = _envflag("SAVE_TOPK_JSON", False)
SKIP_KNN_ON_RAW = False











VARIANTS_DIR = RUN_ROOT / "variants"
WINDOWS_DIR = RUN_ROOT / "windows"
ARTIFACTS_DIR = RUN_ROOT / "artifacts"
RESULTS_DIR = RUN_ROOT / "results"
META_DIR = RUN_ROOT / "meta"
COMMON_SPLITS_DIR = RUN_ROOT / "splits"

for _p in [RUN_ROOT, VARIANTS_DIR, WINDOWS_DIR, ARTIFACTS_DIR, RESULTS_DIR, META_DIR, COMMON_SPLITS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


MODEL_SAVE_DIR = ARTIFACTS_DIR / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


RUN_META_PATH = META_DIR / "run_meta.json"
if not RUN_META_PATH.exists():
    try:
        with open(RUN_META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": DATASET_NAME,
                    "run_id": RUN_ID,
                    "input_ratings_path": INPUT_RATINGS_PATH,
                    "seed": SEED,
                    "seeds": SEEDS,
                    "topk": TOPK,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass





def read_csv_smart(path, usecols=None, dtype=None):

    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, engine="c")
    except Exception:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, engine="python")


try:
    import torch as _torch
    _old_torch_load = _torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _old_torch_load(*args, **kwargs)

    _torch.load = _torch_load_compat

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, get_model, get_trainer

except ImportError:
    _torch = None
    Config = create_dataset = data_preparation = init_seed = get_model = get_trainer = None


def _load_surprise():
    global SVD, KNNBasic, KNNWithMeans, Dataset, Reader, accuracy, train_test_split
    from surprise import SVD, KNNBasic, KNNWithMeans, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split




def _safe_path_no_overwrite(path: Path) -> Path:

    path = Path(path)
    if not path.exists():
        return path
    ts = time.strftime('%Y%m%d_%H%M%S')
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    cand = parent / f"{stem}_{ts}{suffix}"
    i = 2
    while cand.exists():
        cand = parent / f"{stem}_{ts}_{i}{suffix}"
        i += 1
    return cand

def _write_json_no_overwrite(obj, path: Path, indent: int = 2):
    path = _safe_path_no_overwrite(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return str(path)

def _write_csv_no_overwrite(df: pd.DataFrame, path: Path):
    path = _safe_path_no_overwrite(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def quick_stats(frame: pd.DataFrame, name: str = "df"):
    n_rows = len(frame)
    n_users = frame["user_id"].nunique()
    n_items = frame["item_id"].nunique()
    density = n_rows / (n_users * n_items) if n_users and n_items else 0.0
    print(f"[{name}] rows={n_rows:,} | users={n_users:,} | items={n_items:,} | density={density:.8f}")


def recbole_dl_to_df_strict(dl):


    ds = dl.dataset
    uf, itf = ds.uid_field, ds.iid_field

    u_chunks, i_chunks = [], []

    for batch in dl:
        inter = batch
        if isinstance(inter, (tuple, list)):
            inter = inter[0]
        if hasattr(inter, "interaction"):
            inter = inter.interaction

        
        u = inter[uf].detach().cpu().numpy().reshape(-1)
        i = inter[itf].detach().cpu().numpy().reshape(-1)

        
        if "label" in inter:
            lab = inter["label"].detach().cpu().numpy().reshape(-1)
            mask = lab > 0.5
            if mask.any():
                u = u[mask]
                i = i[mask]
            else:
                continue

        u_chunks.append(u)
        i_chunks.append(i)

    if not u_chunks:
        return pd.DataFrame(columns=["user_id", "item_id", "rating"])

    u_all = np.concatenate(u_chunks)
    i_all = np.concatenate(i_chunks)

    u_tok = ds.id2token(uf, u_all)
    i_tok = ds.id2token(itf, i_all)

    out = (
        pd.DataFrame(
            {
                "user_id": [str(x) for x in u_tok],
                "item_id": [str(x) for x in i_tok],
                "rating": 1.0,
            }
        )
        .drop_duplicates(subset=["user_id", "item_id"])
        .reset_index(drop=True)
    )

    return out




def _infer_interaction_columns(columns) -> Tuple[str, str, Optional[str]]:


    cols = list(columns)
    low = {c.lower(): c for c in cols}

    def pick(cands):
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None

    user_col = pick([
        "user_id", "userid", "userId", "userID", "user", "uid", "listener_id", "listenerid"
    ])
    item_col = pick([
        "item_id", "itemid", "itemId", "itemID", "item",
        "movieId", "movie_id",
        "artistId", "artist_id", "artistID", "artist", "artistid",
        "trackId", "track_id", "trackID", "track", "trackid",
        "songId", "song_id", "songID", "song", "songid",
        "stockcode", "stock_code",
    ])
    rating_col = pick([
        "rating", "count", "playcount", "plays", "weight", "listens", "implicit", "score", "quantity"
    ])

    if user_col is None or item_col is None:
        raise RuntimeError(f"Could not infer user/item columns from input columns: {cols}")

    return user_col, item_col, rating_col


def _infer_time_column(columns) -> Optional[str]:

    cols = list(columns)
    low = {c.lower(): c for c in cols}

    candidates = [
        "timestamp", "datetime", "date", "time",
        "event_time", "event_timestamp",
        "created_at", "updated_at",
        "invoice_date", "invoicedate",
    ]
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def interaction_keep_columns(df: pd.DataFrame) -> list:

    cols = [c for c in ["user_id", "item_id", "rating", "timestamp"] if c in df.columns]
    if len(cols) < 3:
        missing = [c for c in ["user_id", "item_id", "rating"] if c not in cols]
        raise RuntimeError(f"Missing required interaction column(s): {missing}")
    return cols


def deduplicate_interactions(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:


    work = df.copy()
    cols = interaction_keep_columns(work)
    work = work[cols].copy()

    if time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = (
            work
            .sort_values(["user_id", "item_id", time_col], kind="mergesort")
            .drop_duplicates(subset=["user_id", "item_id"], keep="last")
            .reset_index(drop=True)
        )
    else:
        work = work.drop_duplicates(subset=["user_id", "item_id"], keep="last").reset_index(drop=True)

    return work


def _coerce_id_series(s: pd.Series, prefer_int: bool = True) -> pd.Series:


    if not prefer_int:
        return s.astype(str)

    try:
        x = pd.to_numeric(s, errors="raise")
        if (x.dropna() % 1).abs().max() > 1e-9:
            return s.astype(str)

        x_int = x.astype("int64", copy=False)
        mn = int(x_int.min()) if len(x_int) else 0
        mx = int(x_int.max()) if len(x_int) else 0
        if mn >= np.iinfo(np.int32).min and mx <= np.iinfo(np.int32).max:
            return x_int.astype("int32", copy=False)
        return x_int
    except Exception:
        return s.astype(str)


def normalize_cols_any(df: pd.DataFrame) -> pd.DataFrame:


    u_col, i_col, r_col = _infer_interaction_columns(df.columns)
    t_col = _infer_time_column(df.columns)

    keep = [u_col, i_col]
    if r_col is not None:
        keep.append(r_col)
    if t_col is not None:
        keep.append(t_col)

    out = df[keep].copy()
    out = out.rename(columns={u_col: "user_id", i_col: "item_id"})
    if r_col is None:
        out["rating"] = 1.0
    else:
        out = out.rename(columns={r_col: "rating"})
        if out["rating"].isna().all():
            out["rating"] = 1.0

    if t_col is not None:
        out = out.rename(columns={t_col: "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    prefer_int = _envflag("PREFER_INT_IDS", True)
    out["user_id"] = _coerce_id_series(out["user_id"], prefer_int=prefer_int)
    out["item_id"] = _coerce_id_series(out["item_id"], prefer_int=prefer_int)

    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(1.0).astype("float32", copy=False)

    return out[interaction_keep_columns(out)]


def read_interactions_csv(path: str) -> pd.DataFrame:


    try:
        cols = list(pd.read_csv(path, nrows=0).columns)
        u_col, i_col, r_col = _infer_interaction_columns(cols)
        t_col = _infer_time_column(cols)
        usecols = [u_col, i_col]
        if r_col is not None:
            usecols.append(r_col)
        if t_col is not None:
            usecols.append(t_col)
        df = read_csv_smart(path, usecols=usecols)
    except Exception:
        df = read_csv_smart(path)
    return normalize_cols_any(df)



def recbole_dataset_to_df(dataset):


    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    inter = dataset.inter_feat
    if inter is None or len(inter) == 0:
        return pd.DataFrame(columns=["user_id", "item_id", "rating"])

    uids = inter[uid_field].detach().cpu().numpy().reshape(-1)
    iids = inter[iid_field].detach().cpu().numpy().reshape(-1)

    raw_uids = dataset.id2token(uid_field, uids)
    raw_iids = dataset.id2token(iid_field, iids)

    
    if "rating" in inter:
        r = inter["rating"].detach().cpu().numpy().reshape(-1).astype(float)
    else:
        r = np.ones(len(uids), dtype=float)

    df = pd.DataFrame(
        {
            "user_id": [str(x) for x in raw_uids],
            "item_id": [str(x) for x in raw_iids],
            "rating": r.astype("float32"),
        }
    )

    
    df = df.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)
    return df



def filter_by_activity_iterative(frame: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:

    df_filt = deduplicate_interactions(normalize_cols_any(frame))
    while True:
        prev_n = len(df_filt)
        user_counts = df_filt["user_id"].value_counts()
        item_counts = df_filt["item_id"].value_counts()
        if min_user > 0:
            active_users = user_counts[user_counts >= min_user].index
            df_filt = df_filt[df_filt["user_id"].isin(active_users)]
        if min_item > 0:
            popular_items = item_counts[item_counts >= min_item].index
            df_filt = df_filt[df_filt["item_id"].isin(popular_items)]
        if len(df_filt) == prev_n:
            break
    return df_filt.reset_index(drop=True)



def save_variant(frame: pd.DataFrame, name: str, out_dir=None):


    force_overwrite = _envflag('FORCE_EXPORT_OVERWRITE', False)
    out_dir = Path(out_dir) if out_dir is not None else VARIANTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"ratings_{name}.csv"
    out_par = out_dir / f"ratings_{name}.parquet"
    if out_csv.exists() and (not force_overwrite):
        print(f"[SKIP] Already exists: {out_csv}")
        return str(out_csv)
    frame.to_csv(out_csv, index=False)
    try:
        frame.to_parquet(out_par, index=False)
    except Exception:
        pass
    quick_stats(frame, name)
    print(f"-> Saved: {out_csv}")
    return str(out_csv)




def gini(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-12
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return np.nan
    p = counts / s
    p = p[p > 0]
    return -np.sum(p * np.log2(p))




def _safe_skew(array_like):

    x = np.asarray(array_like, dtype=float)
    if x.size < 3 or np.all(x == x[0]):
        return np.nan
    try:
        from scipy.stats import skew
        return float(skew(x, bias=False))
    except Exception:
        
        n = x.size
        mean = x.mean()
        s = x.std(ddof=1)
        if s == 0:
            return np.nan
        m3 = np.mean((x - mean) ** 3)
        g1 = m3 / (s ** 3)
        
        return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def activity_skewness(counts):
    c = np.asarray(counts, dtype=float)
    return _safe_skew(c)


def rating_variance(df: pd.DataFrame):
    r = df["rating"].to_numpy(dtype=float, copy=False)
    return float(np.var(r, ddof=1)) if r.size > 1 and not np.all(r == r[0]) else (0.0 if r.size > 0 else np.nan)


def _seems_integer_scale(s: pd.Series) -> bool:
    
    vals = pd.Series(s.dropna().unique())
    if len(vals) == 0:
        return False
    frac = (vals % 1).abs()
    return bool((frac < 1e-9).all())


def compute_binary_threshold(df: pd.DataFrame) -> float:


    if RECBOLE_RATING_THRESHOLD is not None:
        return float(RECBOLE_RATING_THRESHOLD)

    _, rmax = RATING_SCALE
    raw_thr = BINARY_THRESHOLD_FRACTION * rmax

    if _seems_integer_scale(df["rating"]):
        return float(math.ceil(raw_thr - 1e-9))
    return float(raw_thr)


def binarize_for_implicit(df: pd.DataFrame, ensure_min_pos_user: int = IMPLICIT_MIN_POS_PER_USER):


    thr = compute_binary_threshold(df)

    work = normalize_cols_any(df)
    work = work.loc[work["rating"] >= thr].copy()

    
    work = deduplicate_interactions(work)

    if ensure_min_pos_user and ensure_min_pos_user > 0:
        work = work.groupby("user_id").filter(lambda g: len(g) >= ensure_min_pos_user).copy()

    work["rating"] = 1.0
    return work.reset_index(drop=True), thr


def maybe_binarize(df: pd.DataFrame, enabled: bool = True, ensure_min_pos_user: int = IMPLICIT_MIN_POS_PER_USER):


    work = normalize_cols_any(df).copy()

    
    if "rating" not in work.columns:
        work = deduplicate_interactions(work)
        work["rating"] = 1.0
        return work.reset_index(drop=True), None

    rmin = float(work["rating"].min())
    rmax = float(work["rating"].max())

    
    if rmin >= -1e-9 and rmax <= 1.0 + 1e-9:
        work = deduplicate_interactions(work)
        work["rating"] = 1.0
        if ensure_min_pos_user and ensure_min_pos_user > 0:
            work = work.groupby("user_id").filter(lambda g: len(g) >= ensure_min_pos_user).copy()
        return work.reset_index(drop=True), "prebinarized"

    if not enabled:
        work = deduplicate_interactions(work)
        work["rating"] = 1.0
        if ensure_min_pos_user and ensure_min_pos_user > 0:
            work = work.groupby("user_id").filter(lambda g: len(g) >= ensure_min_pos_user).copy()
        return work.reset_index(drop=True), None

    
    bin_df, thr = binarize_for_implicit(work, ensure_min_pos_user=ensure_min_pos_user)
    return bin_df.reset_index(drop=True), thr
def compute_item_pop(df: pd.DataFrame) -> pd.Series:
    
    return df["item_id"].value_counts()


def drop_items_by_pop_percentile(df: pd.DataFrame, top_pct: float = None, bottom_pct: float = None) -> pd.DataFrame:


    pop = compute_item_pop(df)
    n_items = len(pop)
    keep_items = pop.index

    if top_pct is not None and top_pct > 0:
        k_top = max(1, int(round(n_items * top_pct)))
        top_items = pop.sort_values(ascending=False).head(k_top).index
        keep_items = keep_items.difference(top_items)

    if bottom_pct is not None and bottom_pct > 0:
        k_bot = max(1, int(round(n_items * bottom_pct)))
        bot_items = pop.sort_values(ascending=True).head(k_bot).index
        keep_items = keep_items.difference(bot_items)

    df_out = df[df["item_id"].isin(keep_items)].copy()
    return df_out.reset_index(drop=True)


def sample_users_fixed_size(df: pd.DataFrame, target_users: int, seed: int = SEED) -> pd.DataFrame:


    unique_users = df["user_id"].unique()
    n_users = len(unique_users)
    if n_users <= target_users:
        return df.copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    keep_users = rng.choice(unique_users, size=target_users, replace=False)
    out = df[df["user_id"].isin(keep_users)].copy().reset_index(drop=True)

    
    out = out.groupby("user_id").filter(lambda g: len(g) > 0).reset_index(drop=True)
    return out


def sample_users_aligned(raw_df: pd.DataFrame, sparse_df: pd.DataFrame, dense_df: pd.DataFrame,
                         target_users: int, seed: int = SEED):


    common_users = (
        set(raw_df["user_id"].unique())
        & set(sparse_df["user_id"].unique())
        & set(dense_df["user_id"].unique())
    )
    n_common = len(common_users)
    print(f"Aligned sampling: common users before sampling = {n_common:,}")

    if n_common <= target_users:
        raw_out = raw_df[raw_df["user_id"].isin(common_users)].copy()
        sparse_out = sparse_df[sparse_df["user_id"].isin(common_users)].copy()
        dense_out = dense_df[dense_df["user_id"].isin(common_users)].copy()
        return raw_out, sparse_out, dense_out

    rng = np.random.default_rng(seed)
    chosen_users = rng.choice(list(common_users), size=target_users, replace=False)

    raw_out = raw_df[raw_df["user_id"].isin(chosen_users)].copy()
    sparse_out = sparse_df[sparse_df["user_id"].isin(chosen_users)].copy()
    dense_out = dense_df[dense_df["user_id"].isin(chosen_users)].copy()

    return raw_out, sparse_out, dense_out


def sample_items_aligned(df1, df2, df3, target_items=None, seed=SEED):

    
    common_items = (
        set(df1["item_id"].unique())
        & set(df2["item_id"].unique())
        & set(df3["item_id"].unique())
    )
    print(f"Aligned sampling: common items before sampling = {len(common_items):,}")

    
    if target_items is not None and target_items > 0 and len(common_items) > target_items:
        rng = np.random.default_rng(seed)
        chosen_items = set(rng.choice(list(common_items), size=int(target_items), replace=False))
    else:
        chosen_items = common_items

    
    df1_out = df1[df1["item_id"].isin(chosen_items)].copy()
    df2_out = df2[df2["item_id"].isin(chosen_items)].copy()
    df3_out = df3[df3["item_id"].isin(chosen_items)].copy()

    print(f"Aligned sampling: final items = {len(chosen_items):,}")
    return df1_out, df2_out, df3_out


def _variant_name_from_csv_path(path_like) -> str:
    stem = Path(path_like).stem
    return stem[len("ratings_"):] if stem.startswith("ratings_") else stem


def common_split_paths(variant_name: str) -> dict:
    split_dir = COMMON_SPLITS_DIR / str(variant_name)
    return {
        "dir": split_dir,
        "train_csv": split_dir / "train.csv",
        "test_csv": split_dir / "test.csv",
        "meta_json": split_dir / "meta.json",
    }


def split_user_loo(df: pd.DataFrame, seed: int = SEED, shuffle: bool = True):


    split_mode = os.getenv("COMMON_SPLIT_MODE", "auto").strip().lower()
    time_col = os.getenv("COMMON_SPLIT_TIME_COL", "timestamp").strip() or "timestamp"

    work = normalize_cols_any(df)

    has_time = (
        time_col in work.columns
        and work[time_col].notna().any()
    ) or (
        "timestamp" in work.columns
        and work["timestamp"].notna().any()
    )

    if time_col not in work.columns and "timestamp" in work.columns:
        time_col = "timestamp"

    use_chrono = (
        (split_mode == "chrono_loo")
        or (split_mode == "auto" and has_time)
    )

    if use_chrono:
        if time_col not in work.columns:
            raise RuntimeError(
                "Chronological split was requested, but no timestamp column is available."
            )

        work = work.dropna(subset=[time_col]).copy()
        work = deduplicate_interactions(work, time_col=time_col)
        work = work.sort_values(["user_id", time_col, "item_id"], kind="mergesort").reset_index(drop=True)

        train_parts, test_parts = [], []
        for _, g in work.groupby("user_id", sort=False):
            g = g.copy().reset_index(drop=True)
            n = len(g)
            if n == 0:
                continue
            if n >= 2:
                train_parts.append(g.iloc[:-1].copy())
                test_parts.append(g.iloc[[-1]].copy())
            else:
                train_parts.append(g.copy())

        empty = work.iloc[:0].copy()
        train_df = pd.concat(train_parts, ignore_index=True) if train_parts else empty.copy()
        test_df = pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy()
        return train_df, test_df

    work = deduplicate_interactions(work)

    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []

    for _, g in work.groupby("user_id", sort=False):
        g = g.copy().reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue

        if shuffle and n > 1:
            idx = np.arange(n)
            rng.shuffle(idx)
            g = g.iloc[idx].reset_index(drop=True)

        if n >= 2:
            train_parts.append(g.iloc[:-1].copy())
            test_parts.append(g.iloc[[-1]].copy())
        else:
            train_parts.append(g.copy())

    empty = work.iloc[:0].copy()
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else empty.copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy()
    return train_df, test_df


def save_common_split_for_variant_df(
    df: pd.DataFrame,
    variant_name: str,
    seed: int = SEED,
    shuffle: bool = True,
    force: bool = False,
    source_csv: Optional[str] = None,
):

    paths = common_split_paths(variant_name)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if (not force) and paths["train_csv"].exists() and paths["test_csv"].exists() and paths["meta_json"].exists():
        print(f"[SPLIT][SKIP] Already exists: {variant_name}")
        return paths

    dfv = normalize_cols_any(df)
    dfv = dfv[interaction_keep_columns(dfv)].copy()
    train_df, test_df = split_user_loo(dfv, seed=seed, shuffle=shuffle)

    train_df.to_csv(paths["train_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)
    effective_split_type = (
        "chrono_loo"
        if ("timestamp" in dfv.columns and dfv["timestamp"].notna().any()
            and os.getenv("COMMON_SPLIT_MODE", "auto").strip().lower() != "random_loo")
        else "random_loo"
    )

    meta = {
        "variant": str(variant_name),
        "source_csv": str(source_csv) if source_csv else None,
        "split_type": effective_split_type,
        "has_validation": False,
        "shuffle_within_user": bool(shuffle),
        "seed": int(seed),
        "time_col": os.getenv("COMMON_SPLIT_TIME_COL", "timestamp").strip() or "timestamp",
        "rows_total": int(len(dfv)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "users_total": int(dfv["user_id"].nunique()),
        "users_in_train": int(train_df["user_id"].nunique()),
        "users_in_test": int(test_df["user_id"].nunique()),
        "items_total": int(dfv["item_id"].nunique()),
        "items_in_train": int(train_df["item_id"].nunique()),
        "items_in_test": int(test_df["item_id"].nunique()),
    }
    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[SPLIT] {variant_name}: train_rows={len(train_df):,} | test_rows={len(test_df):,} | "
        f"train_users={train_df['user_id'].nunique():,} | test_users={test_df['user_id'].nunique():,}"
    )
    return paths


def save_common_split_for_variant_csv(
    csv_path: str,
    seed: int = SEED,
    shuffle: bool = True,
    force: bool = False,
):

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Variant CSV not found: {csv_path}")

    variant_name = _variant_name_from_csv_path(csv_path)
    dfv = read_interactions_csv(str(csv_path))
    return save_common_split_for_variant_df(
        df=dfv,
        variant_name=variant_name,
        seed=seed,
        shuffle=shuffle,
        force=force,
        source_csv=str(csv_path),
    )


def build_common_splits_for_all_variants(
    variants_dir: Optional[str] = None,
    seed: int = SEED,
    shuffle: bool = False,
    force: bool = False,
):

    base_dir = Path(variants_dir) if variants_dir is not None else Path(VARIANTS_DIR)
    csv_paths = sorted(base_dir.glob("ratings_*.csv"))
    if not csv_paths:
        print("[SPLIT] No variant CSV files found.")
        return []

    out = []
    print(f"[SPLIT] Building common LOO splits for {len(csv_paths)} variant file(s)...")
    for csv_path in csv_paths:
        out.append(
            save_common_split_for_variant_csv(
                csv_path=str(csv_path),
                seed=seed,
                shuffle=shuffle,
                force=force,
            )
        )
    return out


def load_variant_split(
    variant_name: str,
    create_if_missing: bool = False,
    seed: int = SEED,
    shuffle: bool = False,
):


    paths = common_split_paths(variant_name)

    if create_if_missing and (not paths["train_csv"].exists() or not paths["test_csv"].exists()):
        source_csv = Path(VARIANTS_DIR) / f"ratings_{variant_name}.csv"
        if not source_csv.exists():
            raise FileNotFoundError(f"Variant source CSV not found for split creation: {source_csv}")
        save_common_split_for_variant_csv(
            csv_path=str(source_csv),
            seed=seed,
            shuffle=shuffle,
            force=False,
        )

    if not paths["train_csv"].exists() or not paths["test_csv"].exists():
        raise FileNotFoundError(
            f"Missing split files for variant='{variant_name}'. Expected: {paths['train_csv']} and {paths['test_csv']}"
        )

    train_df = read_interactions_csv(str(paths["train_csv"]))
    test_df = read_interactions_csv(str(paths["test_csv"]))

    meta = None
    if paths["meta_json"].exists():
        try:
            with open(paths["meta_json"], "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None

    return train_df, test_df, meta


def split_user_random_rs(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed=SEED):


    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1.0"
    rng = np.random.default_rng(seed)
    parts = []

    for uid, g in df.groupby("user_id", sort=False):
        n = len(g)
        if n == 0:
            continue

        idx = np.arange(n)
        rng.shuffle(idx)

        n_train = int(n * ratios[0])
        n_valid = int(n * ratios[1])

        
        if n >= 2:
            n_train = max(1, n_train)
        n_train = min(n_train, n)  

        
        n_valid = max(0, min(n_valid, n - n_train - 1)) if n - n_train >= 2 else 0

        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]

        parts.append((g.iloc[train_idx], g.iloc[valid_idx], g.iloc[test_idx]))

    train_df = pd.concat([p[0] for p in parts], ignore_index=True) if parts else df.iloc[:0].copy()
    valid_df = pd.concat([p[1] for p in parts], ignore_index=True) if parts else df.iloc[:0].copy()
    test_df = pd.concat([p[2] for p in parts], ignore_index=True) if parts else df.iloc[:0].copy()
    return train_df, valid_df, test_df


def df_to_surprise_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = train_df.copy()
    te = test_df.copy()

    
    tr["user_id"] = tr["user_id"].astype(str)
    tr["item_id"] = tr["item_id"].astype(str)
    te["user_id"] = te["user_id"].astype(str)
    te["item_id"] = te["item_id"].astype(str)

    reader = _reader_for_df(tr)
    train_data = Dataset.load_from_df(tr[["user_id", "item_id", "rating"]], reader)
    trainset = train_data.build_full_trainset()
    testset = list(te[["user_id", "item_id", "rating"]].itertuples(index=False, name=None))
    return trainset, testset


def log_sparsity_df(df: pd.DataFrame, tag: str):
    ucnt = df["user_id"].value_counts()
    icnt = df["item_id"].value_counts()
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    density = len(df) / (n_users * n_items) if (n_users and n_items) else 0.0
    sparsity = 1.0 - density
    u_gini = gini(ucnt)
    i_gini = gini(icnt)
    u_H = shannon_entropy(ucnt)
    i_H = shannon_entropy(icnt)
    u_skew = activity_skewness(ucnt)
    i_skew = activity_skewness(icnt)
    
    r_var = rating_variance(df) if "rating" in df.columns else np.nan

    print(
        f"[{tag}] rows={len(df):,} | users={n_users:,} | items={n_items:,} | "
        f"density={density:.8f} | sparsity={sparsity:.8f} | "
        f"user_gini={u_gini:.4f} | item_gini={i_gini:.4f} | "
        f"user_skew={u_skew:.3f} | item_skew={i_skew:.3f} | "
        f"user_H={u_H:.2f} | item_H={i_H:.2f} | rating_var={r_var:.4f}"
    )


def _log_pos_stats(df_bin: pd.DataFrame, tag: str):
    n_users = df_bin["user_id"].nunique()
    pos_per_user = df_bin.groupby("user_id")["item_id"].nunique()
    print(f"[{tag}] users={n_users:,} | avg positives/user={pos_per_user.mean():.2f} | median={pos_per_user.median():.0f}")


def sparsity_report(df: pd.DataFrame, name: str):

    return sparsity_report_from_df(df, name)


def sparsity_report_from_trainset(trainset, name: str, return_dict: bool = False):
    ucnt = defaultdict(int)
    icnt = defaultdict(int)
    n = 0
    ratings_list = []
    for u_inner, i_inner, r in trainset.all_ratings():
        uid = trainset.to_raw_uid(u_inner)
        iid = trainset.to_raw_iid(i_inner)
        ucnt[uid] += 1
        icnt[iid] += 1
        n += 1
        ratings_list.append(r)

    n_users, n_items = len(ucnt), len(icnt)
    density = n / (n_users * n_items) if (n_users and n_items) else 0.0
    sparsity = 1.0 - density

    user_counts = np.fromiter(ucnt.values(), dtype=float)
    item_counts = np.fromiter(icnt.values(), dtype=float)

    user_g = gini(user_counts)
    item_g = gini(item_counts)
    user_H = shannon_entropy(user_counts)
    item_H = shannon_entropy(item_counts)
    user_sk = activity_skewness(user_counts)
    item_sk = activity_skewness(item_counts)
    r_var = rating_variance(pd.DataFrame({"rating": ratings_list})) if ratings_list else np.nan

    print(
        f"[{name}] rows={n:,} | users={n_users:,} | items={n_items:,} | "
        f"density={density:.8f} | sparsity={sparsity:.8f} | "
        f"user_gini={user_g:.4f} | item_gini={item_g:.4f} | "
        f"user_skew={user_sk:.3f} | item_skew={item_sk:.3f} | "
        f"user_H={user_H:.2f} | item_H={item_H:.2f} | rating_var={r_var:.4f}"
    )

    if return_dict:
        return {
            "name": name,
            "rows": n,
            "users": n_users,
            "items": n_items,
            "density": float(density),
            "sparsity": float(sparsity),
            "user_gini": float(user_g),
            "item_gini": float(item_g),
            "user_skew": float(user_sk),
            "item_skew": float(item_sk),
            "user_H": float(user_H),
            "item_H": float(item_H),
            "rating_var": float(r_var) if r_var == r_var else np.nan,
        }


def sparsity_report_from_df(df: pd.DataFrame, name: str):
    ucnt = df["user_id"].value_counts()
    icnt = df["item_id"].value_counts()

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    density = len(df) / (n_users * n_items) if (n_users and n_items) else 0.0
    sparsity = 1.0 - density

    u_gini = gini(ucnt)
    i_gini = gini(icnt)
    u_H = shannon_entropy(ucnt)
    i_H = shannon_entropy(icnt)
    u_skew = activity_skewness(ucnt)
    i_skew = activity_skewness(icnt)
    r_var = rating_variance(df)

    print(
        f"[{name}] rows={len(df):,} | users={n_users:,} | items={n_items:,} | "
        f"density={density:.8f} | sparsity={sparsity:.8f} | "
        f"user_gini={u_gini:.4f} | item_gini={i_gini:.4f} | "
        f"user_skew={u_skew:.3f} | item_skew={i_skew:.3f} | "
        f"user_H={u_H:.2f} | item_H={i_H:.2f} | rating_var={r_var:.4f}"
    )




def _reader_for_df(df: pd.DataFrame) -> "Reader":
    
    if SURPRISE_IMPLICIT:
        return Reader(rating_scale=(0, 1))
    return Reader(rating_scale=RATING_SCALE)


def df_to_surprise(df: pd.DataFrame):
    reader = _reader_for_df(df)
    return Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)


def augment_train_with_negatives(
    train_pos_df: pd.DataFrame,
    all_items: np.ndarray,
    neg_per_pos: int = 4,
    seed: int = SEED,
    max_neg_per_user: int = None,
) -> pd.DataFrame:


    rng = np.random.default_rng(seed)

    
    pos = train_pos_df[["user_id", "item_id"]].drop_duplicates().copy()
    pos["rating"] = 1.0

    all_items = np.asarray(all_items)
    if all_items.size == 0 or len(pos) == 0:
        return pos

    neg_parts = []
    for uid, g in pos.groupby("user_id", sort=False):
        pos_items = set(g["item_id"].tolist())
        n_pos = len(pos_items)
        if n_pos == 0:
            continue

        n_neg = int(neg_per_pos * n_pos)
        if max_neg_per_user is not None:
            n_neg = min(n_neg, int(max_neg_per_user))
        if n_neg <= 0:
            continue

        
        mask = np.array([it not in pos_items for it in all_items], dtype=bool)
        cand = all_items[mask]
        if cand.size == 0:
            continue

        take = min(n_neg, cand.size)
        sampled = rng.choice(cand, size=take, replace=False)

        neg_df = pd.DataFrame(
            {
                "user_id": [uid] * take,
                "item_id": sampled,
                "rating": 0.0,
            }
        )
        neg_parts.append(neg_df)

    if neg_parts:
        neg = pd.concat(neg_parts, ignore_index=True)
        out = pd.concat([pos, neg], ignore_index=True)
    else:
        out = pos

    return out


def build_user_histories(trainset):
    user_items = defaultdict(set)
    for u_inner, i_inner, r in trainset.all_ratings():
        if r <= 0.5:
            continue
        uid = trainset.to_raw_uid(u_inner)
        iid = trainset.to_raw_iid(i_inner)
        user_items[uid].add(iid)
    return user_items


def build_item_user_sets(trainset):
    item_users = defaultdict(set)
    for u_inner, i_inner, r in trainset.all_ratings():
        if r <= 0.5:
            continue
        uid = trainset.to_raw_uid(u_inner)
        iid = trainset.to_raw_iid(i_inner)
        item_users[iid].add(uid)
    return item_users


class ImplicitItemKNN:



    def __init__(
        self,
        topk_neighbors: int = 200,
        shrink: float = 10.0,
        min_common: int = 2,
        max_items_per_user: int = 200,
        seed: int = SEED,
    ):
        self.topk_neighbors = int(topk_neighbors)
        self.shrink = float(shrink)
        self.min_common = int(min_common)
        self.max_items_per_user = int(max_items_per_user)
        self.seed = int(seed)

        self.trainset = None
        self.user_histories = None
        self.item_users = None
        self._item_deg = None
        self._neighbors = None

    def fit(self, trainset):
        self.trainset = trainset
        self.user_histories = build_user_histories(trainset)
        self.item_users = build_item_user_sets(trainset)

        
        self._item_deg = {iid: len(u_set) for iid, u_set in self.item_users.items()}

        
        co = defaultdict(Counter)
        rng = np.random.RandomState(self.seed)
        for _, items_set in self.user_histories.items():
            items = list(items_set)
            if len(items) > self.max_items_per_user:
                items = rng.choice(items, size=self.max_items_per_user, replace=False).tolist()

            n = len(items)
            for a in range(n):
                ia = items[a]
                for b in range(a + 1, n):
                    ib = items[b]
                    co[ia][ib] += 1
                    co[ib][ia] += 1

        
        neighbors = {}
        for ia, cnts in co.items():
            da = self._item_deg.get(ia, 0)
            if da <= 0:
                continue
            na = math.sqrt(da)

            sims = []
            for ib, c in cnts.items():
                if c < self.min_common:
                    continue
                db = self._item_deg.get(ib, 0)
                if db <= 0:
                    continue
                sim = c / (na * math.sqrt(db))
                if self.shrink > 0:
                    sim *= c / (c + self.shrink)
                sims.append((ib, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            if self.topk_neighbors > 0:
                sims = sims[: self.topk_neighbors]
            neighbors[ia] = sims

        self._neighbors = neighbors
        return self

    def score_candidates(self, uid: str, candidates: list[str]) -> list[float]:

        hist = self.user_histories.get(uid)
        if not hist:
            return [0.0] * len(candidates)

        cand_index = {iid: idx for idx, iid in enumerate(candidates)}
        scores = np.zeros(len(candidates), dtype=np.float32)

        
        for j in hist:
            for i, sim in self._neighbors.get(j, []):
                idx = cand_index.get(i)
                if idx is not None:
                    scores[idx] += sim

        return scores.tolist()

    def predict(self, uid, iid, verbose=False):
        
        class _Pred:
            def __init__(self, est):
                self.est = est

        est = float(self.score_candidates(uid, [iid])[0])
        return _Pred(est)


def sample_candidates(all_items, exclude_set, n_sample, rng):
    if n_sample is None or n_sample <= 0:
        return [iid for iid in all_items if iid not in exclude_set]

    need = int(n_sample)
    if need <= 0:
        return []

    
    avail = max(0, len(all_items) - len(exclude_set))
    need = min(need, avail)
    if need <= 0:
        return []

    picked = set()
    all_arr = np.asarray(all_items)

    max_tries = need * 50
    tries = 0

    while len(picked) < need and tries < max_tries:
        batch_size = min(need - len(picked), 256, len(all_arr))
        if batch_size <= 0:
            break

        batch = rng.choice(all_arr, size=batch_size, replace=False)
        for iid in batch:
            if iid not in exclude_set and iid not in picked:
                picked.add(iid)
                if len(picked) >= need:
                    break
        tries += 1

    return list(picked)




def ndcg_at_k(gt_items, ranked_items, k=10):
    dcg = 0.0
    for rank, iid in enumerate(ranked_items[:k], start=1):
        if iid in gt_items:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_rel_count = min(len(gt_items), k)
    if ideal_rel_count == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_rel_count + 1))
    return dcg / idcg


def recall_at_k(gt_items, ranked_items, k=10):
    if len(gt_items) == 0:
        return 0.0
    hits = sum(1 for iid in ranked_items[:k] if iid in gt_items)
    return hits / len(gt_items)


def precision_at_k(gt_items, ranked_items, k=10):
    if k == 0:
        return 0.0
    denom = min(k, len(ranked_items))
    if denom == 0:
        return 0.0
    hits = sum(1 for iid in ranked_items[:k] if iid in gt_items)
    return hits / denom


def mrr_at_k(gt_items, ranked_items, k=10):
    for rank, iid in enumerate(ranked_items[:k], start=1):
        if iid in gt_items:
            return 1.0 / rank
    return 0.0


def epc_at_k(ranked_items, pop_counts, n_users, k=10, discounted=False):


    if n_users <= 0 or not ranked_items:
        return 0.0
    weights = []
    values = []
    for r, iid in enumerate(ranked_items[:k], start=1):
        p_seen = pop_counts.get(iid, 0) / n_users
        comp = 1.0 - p_seen
        if discounted:
            w = 1.0 / np.log2(r + 1)
            weights.append(w)
            values.append(comp * w)
        else:
            values.append(comp)
    if discounted:
        return float(np.sum(values) / (np.sum(weights) + 1e-12)) if weights else 0.0
    return float(np.mean(values)) if values else 0.0


def compute_popularity(trainset):
    
    item_cnt = defaultdict(int)
    for u_inner, i_inner, r in trainset.all_ratings():
        if r <= 0.5:
            continue
        iid = trainset.to_raw_iid(i_inner)
        item_cnt[iid] += 1
    return dict(item_cnt)


def arp_at_k(ranked_items, pop_counts, k=10, normalize=False, n_users=None):


    top = ranked_items[:k]
    if not top:
        return 0.0
    vals = [pop_counts.get(iid, 0) for iid in top]
    if normalize:
        if not n_users or n_users <= 0:
            raise ValueError("n_users must be provided when normalize=True.")
        vals = [v / n_users for v in vals]
    return float(np.mean(vals))


def build_long_tail_set(pop_counts, pop_ratio=0.8):


    if not pop_counts:
        return set()

    total = float(sum(pop_counts.values()))
    if total <= 0:
        return set()

    limit = total * float(pop_ratio)
    cum = 0.0
    head = set()

    for iid, c in sorted(pop_counts.items(), key=lambda x: x[1], reverse=True):
        head.add(iid)
        cum += float(c)
        if cum >= limit:
            break

    return set(pop_counts.keys()) - head

def efd_at_k(ranked_items, pop_counts, k=10, discounted=True):


    top = ranked_items[:k]
    if not top or not pop_counts:
        return 0.0

    norm = float(sum(pop_counts.values()))
    if norm <= 0:
        return 0.0

    min_pop = min(pop_counts.values())
    default_nov = -math.log2(float(min_pop) / norm)

    vals, weights = [], []
    for r, iid in enumerate(top, start=1):
        c = pop_counts.get(iid, 0)
        nov = default_nov if c <= 0 else -math.log2(float(c) / norm)
        if discounted:
            w = 1.0 / np.log2(r + 1)
            weights.append(w)
            vals.append(nov * w)
        else:
            vals.append(nov)

    if discounted:
        return float(np.sum(vals) / (np.sum(weights) + 1e-12)) if weights else 0.0
    return float(np.mean(vals)) if vals else 0.0


def lt_share_at_k(ranked_items, long_tail_set, k=10):
    
    top = ranked_items[:k]
    if not top:
        return 0.0
    return float(sum(1 for iid in top if iid in long_tail_set) / len(top))


def aplt_at_k(ranked_items, long_tail_set, pop_counts, n_users, k=10, normalize=True):


    lt = [iid for iid in ranked_items[:k] if iid in long_tail_set]
    if not lt:
        return 0.0
    vals = [pop_counts.get(iid, 0) for iid in lt]
    if normalize and n_users > 0:
        vals = [v / n_users for v in vals]
    return float(np.mean(vals))


def epc_at_k_rel(gt_items, ranked_items, pop_counts, n_users, k=10, discounted=True):


    top = ranked_items[:k]
    if not top or n_users <= 0:
        return 0.0

    vals = []
    weights = []

    for r, iid in enumerate(top, start=1):
        rel = 1.0 if iid in gt_items else 0.0
        p_seen = pop_counts.get(iid, 0) / n_users
        nov = 1.0 - p_seen

        if discounted:
            w = 1.0 / np.log2(r + 1)
            weights.append(w)
            vals.append(rel * nov * w)
        else:
            vals.append(rel * nov)

    if discounted:
        return float(np.sum(vals) / (np.sum(weights) + 1e-12)) if weights else 0.0
    return float(np.mean(vals)) if vals else 0.0


def efd_at_k_rel(gt_items, ranked_items, pop_counts, k=10, discounted=True):


    top = ranked_items[:k]
    if not top or not pop_counts:
        return 0.0

    norm = float(sum(pop_counts.values()))
    if norm <= 0:
        return 0.0

    
    min_pop = min(pop_counts.values()) if pop_counts else 1
    default_nov = -math.log2(float(min_pop) / norm)

    vals = []
    weights = []

    for r, iid in enumerate(top, start=1):
        rel = 1.0 if iid in gt_items else 0.0
        c = pop_counts.get(iid, None)
        nov = default_nov if (c is None or c <= 0) else -math.log2(float(c) / norm)

        if discounted:
            w = 1.0 / np.log2(r + 1)
            weights.append(w)
            vals.append(rel * nov * w)
        else:
            vals.append(rel * nov)

    if discounted:
        return float(np.sum(vals) / (np.sum(weights) + 1e-12)) if weights else 0.0
    return float(np.mean(vals)) if vals else 0.0

def ils_diversity(rec_items, item_users):
    L = len(rec_items)
    if L <= 1:
        return 0.0, 1.0
    sims = []
    for a in range(L):
        ia = rec_items[a]
        Ua = item_users.get(ia, set())
        for b in range(a + 1, L):
            ib = rec_items[b]
            Ub = item_users.get(ib, set())
            if len(Ua) == 0 or len(Ub) == 0:
                sim = 0.0
            else:
                sim = len(Ua & Ub) / math.sqrt(len(Ua) * len(Ub))
            sims.append(sim)
    ils = float(np.mean(sims)) if sims else 0.0
    return ils, 1.0 - ils


def filter_test_to_train(testset, trainset):
    train_items = {trainset.to_raw_iid(i) for i in trainset.all_items()}
    train_users = {trainset.to_raw_uid(u) for u in trainset.all_users()}
    return [(u, i, r) for (u, i, r) in testset if (i in train_items and u in train_users)]


def evaluate_ranking(
    algo,
    trainset,
    testset,
    topk=TOPK,
    save_ranked=False,
    model_name="model",
    variant_name="variant",
    save_path=EXPORT_DIR,
    neg_sample=NEGATIVE_CANDIDATE_SAMPLE,  
    seed=SEED,  
):
    if (not neg_sample or (isinstance(neg_sample, (int, float)) and neg_sample <= 0)):
        print(f"Ranking eval = FULL-SORT, TOPK={topk}")
    elif isinstance(neg_sample, str) and neg_sample == "match_pos":
        print(f"Ranking eval = CANDIDATE SAMPLING (per-user 1:1), TOPK={topk}")
    else:
        print(f"Ranking eval = CANDIDATE SAMPLING (n={neg_sample}), TOPK={topk}")

    
    all_items_raw = sorted({trainset.to_raw_iid(i) for i in trainset.all_items()})
    user_hist = build_user_histories(trainset)
    item_users = build_item_user_sets(trainset) if COMPUTE_ILS else None

    
    testset = filter_test_to_train(testset, trainset)

    gt = defaultdict(set)
    for uid, iid, _ in testset:
        gt[uid].add(iid)

    
    
    if len(gt) == 0:
        print(f"[WARN] Empty usable testset after filtering to trainset. model={model_name} variant={variant_name} seed={seed}")
        catalog_n = len(all_items_raw)
        out = {
            f"Precision@{topk}": float('nan'),
            f"Recall@{topk}": float('nan'),
            f"NDCG@{topk}": float('nan'),
            f"MRR@{topk}": float('nan'),
            f"Coverage@{topk}": float('nan'),
            "CatalogItems": float(catalog_n),
            f"UniqueRecItems@{topk}": float('nan'),
            f"ItemCV@{topk}": float('nan'),
            f"UserCoverage@{topk}": float('nan'),
            f"EEL@{topk}": float('nan'),
            "RecGini": float('nan'),
            "RecEntropy": float('nan'),
            f"1-Gini@{topk}": float('nan'),
            f"ARP@{topk}": float('nan'),
            f"APLT@{topk}": float('nan'),
            f"LTShare@{topk}": float('nan'),
            f"EPC@{topk}": float('nan'),
            f"EFD@{topk}": float('nan'),
            f"EPC_rel@{topk}": float('nan'),
            f"EFD_rel@{topk}": float('nan'),
            "ILS": float('nan'),
            "Diversity": float('nan'),
        }
        if save_ranked:
            
            _write_json_no_overwrite({}, Path(save_path) / f"topk_{variant_name}_{model_name}_seed{seed}.json")
        return out

    pop_counts = compute_popularity(trainset)

    long_tail = build_long_tail_set(pop_counts, pop_ratio=0.8)

    arp_vals, aplt_vals, lt_share_vals = [], [], []
    epc_vals, efd_vals = [], []
    epc_rel_vals, efd_rel_vals = [], []

    
    epc_no_rel_vals = []


    rec_items_global = set()
    rec_freq = Counter()

    recalls, ndcgs, precisions, mrrs = [], [], [], []
    epc_vals = []
    ils_vals, div_vals = [], []
    user_hit_flags = []
    user_topk = {}

    rng = np.random.default_rng(seed)

    for uid in sorted(gt.keys()):
        user_exclude = user_hist.get(uid, set())

        
        if isinstance(neg_sample, str) and neg_sample == "match_pos":
            n_neg = len(gt[uid])
        else:
            n_neg = neg_sample if neg_sample else 0

        negatives = sample_candidates(all_items_raw, user_exclude | gt[uid], n_neg, rng)

        if hasattr(algo, "score_candidates"):
            candidates = list(dict.fromkeys(negatives + list(gt[uid])))
        else:
            candidates = list(set(negatives) | set(gt[uid]))
            rng.shuffle(candidates)

        if hasattr(algo, "score_candidates"):
            scores = algo.score_candidates(uid, candidates)
            preds = list(zip(candidates, scores))
        else:
            preds = []
            err_count = 0
            max_err_print = 10
            for iid in candidates:
                try:
                    est = algo.predict(uid, iid, verbose=False).est
                except Exception as e:
                    if err_count < max_err_print:
                        import traceback
                        print(f"[PRED-ERR] model={model_name} variant={variant_name} uid={uid} iid={iid} err={repr(e)}")
                        traceback.print_exc()
                    err_count += 1
                    est = 0.0
                preds.append((iid, est))

            if err_count > 0:
                print(f"[PRED-ERR-SUMMARY] model={model_name} variant={variant_name} total_errors={err_count}")

        preds.sort(key=lambda x: x[1], reverse=True)
        ranked = [iid for iid, _ in preds[:topk]]

        for iid in ranked:
            rec_freq[iid] += 1
        rec_items_global.update(ranked)

        recalls.append(recall_at_k(gt[uid], ranked, k=topk))
        ndcgs.append(ndcg_at_k(gt[uid], ranked, k=topk))
        precisions.append(precision_at_k(gt[uid], ranked, k=topk))
        mrrs.append(mrr_at_k(gt[uid], ranked, k=topk))
        user_hit_flags.append(int(any(i in gt[uid] for i in ranked)))

        arp_vals.append(arp_at_k(ranked, pop_counts, k=topk))

        
        aplt_vals.append(aplt_at_k(ranked, long_tail, pop_counts, trainset.n_users, k=topk, normalize=True))
        lt_share_vals.append(lt_share_at_k(ranked, long_tail, k=topk))

        
        epc_vals.append(epc_at_k(ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_vals.append(efd_at_k(ranked, pop_counts, k=topk, discounted=True))

        
        epc_rel_vals.append(epc_at_k_rel(gt[uid], ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_rel_vals.append(efd_at_k_rel(gt[uid], ranked, pop_counts, k=topk, discounted=True))

        if COMPUTE_ILS and item_users is not None:
            ils, div = ils_diversity(ranked, item_users)
            ils_vals.append(ils)
            div_vals.append(div)

        if save_ranked:
            user_topk[uid] = ranked

    if save_ranked:
        filename = f"top{topk}_{model_name}_{variant_name}.json"
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in user_topk.items()}, f)
        print(f"Saved Top-{topk} recommendations to {out_path}")

    catalog_items = len(all_items_raw)
    unique_rec_items = len(rec_items_global)
    coverage = (unique_rec_items / catalog_items) if catalog_items else 0.0
    rec_counts = list(rec_freq.values())

    
    test_item_users = defaultdict(set)
    for uid, items in gt.items():
        for iid in items:
            test_item_users[iid].add(uid)
    U_test = len(gt)
    eel_terms = []
    all_items_test = set(test_item_users.keys()) | set(rec_freq.keys())
    for iid in all_items_test:
        rec_share = rec_freq.get(iid, 0) / max(U_test, 1)
        test_share = len(test_item_users.get(iid, set())) / max(U_test, 1)
        eel_terms.append(abs(rec_share - test_share))
    eel_value = float(np.mean(eel_terms)) if eel_terms else 0.0

    catalog_items = len(all_items_raw)
    unique_rec_items = len(rec_items_global)
    coverage = (unique_rec_items / catalog_items) if catalog_items else 0.0

    
    rec_counts_full = [rec_freq.get(iid, 0) for iid in all_items_raw]
    rec_gini = gini(rec_counts_full) if catalog_items else np.nan

    
    rec_entropy = shannon_entropy([c for c in rec_counts_full if c > 0])

    
    mean_rec = float(np.mean(rec_counts_full)) if rec_counts_full else 0.0
    std_rec = float(np.std(rec_counts_full)) if rec_counts_full else 0.0
    item_cv = float(std_rec / (mean_rec + 1e-12)) if mean_rec > 0 else 0.0

    out = {
    f"Precision@{topk}": float(np.mean(precisions)) if precisions else 0.0,
    f"Recall@{topk}": float(np.mean(recalls)) if recalls else 0.0,
    f"NDCG@{topk}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    f"MRR@{topk}": float(np.mean(mrrs)) if mrrs else 0.0,

    f"Coverage@{topk}": float(coverage),
    "CatalogItems": int(catalog_items),
    f"UniqueRecItems@{topk}": int(unique_rec_items),
    f"ItemCV@{topk}": float(item_cv),
    f"UserCoverage@{topk}": float(np.mean(user_hit_flags)) if user_hit_flags else 0.0,
    }

    if COMPUTE_BEYOND_ACCURACY:
        out.update({
            f"EEL@{topk}": float(eel_value),

            "RecGini": float(rec_gini) if rec_gini == rec_gini else np.nan,
            "RecEntropy": float(rec_entropy) if rec_entropy == rec_entropy else np.nan,
            f"1-Gini@{topk}": float(1.0 - rec_gini) if rec_gini == rec_gini else np.nan,

            f"ARP@{topk}": float(np.mean(arp_vals)) if arp_vals else 0.0,
            f"APLT@{topk}": float(np.mean(aplt_vals)) if aplt_vals else 0.0,
            f"LTShare@{topk}": float(np.mean(lt_share_vals)) if lt_share_vals else 0.0,

            f"EPC@{topk}": float(np.mean(epc_vals)) if epc_vals else 0.0,
            f"EFD@{topk}": float(np.mean(efd_vals)) if efd_vals else 0.0,
            f"EPC_rel@{topk}": float(np.mean(epc_rel_vals)) if epc_rel_vals else 0.0,
            f"EFD_rel@{topk}": float(np.mean(efd_rel_vals)) if efd_rel_vals else 0.0,
        })
    else:
        out.update({
            f"EEL@{topk}": 0.0,
            "RecGini": 0.0,
            "RecEntropy": 0.0,
            f"1-Gini@{topk}": 0.0,
            f"ARP@{topk}": 0.0,
            f"APLT@{topk}": 0.0,
            f"LTShare@{topk}": 0.0,
            f"EPC@{topk}": 0.0,
            f"EFD@{topk}": 0.0,
            f"EPC_rel@{topk}": 0.0,
            f"EFD_rel@{topk}": 0.0,
        })

    if COMPUTE_ILS and ils_vals:
        out["ILS"] = float(np.mean(ils_vals))
        out["Diversity"] = float(np.mean(div_vals)) if div_vals else 0.0
    else:
        out["ILS"] = 0.0
        out["Diversity"] = 0.0

    if save_ranked:
        
        try:
            out_file = Path(save_path) / f"topk_{variant_name}_{model_name}_seed{seed}.json"
            _write_json_no_overwrite({str(k): v for k, v in user_topk.items()}, out_file)
        except Exception as e:
            print(f"[WARN] Could not save Top-K JSON: {repr(e)}")

    return out


def evaluate_ranking_from_saved_topk(
    user_topk: dict,
    trainset,
    testset,
    topk=TOPK,
    model_name="model",
    variant_name="variant",
    seed=SEED,
):


    all_items_raw = sorted({trainset.to_raw_iid(i) for i in trainset.all_items()})
    item_users = build_item_user_sets(trainset) if COMPUTE_ILS else None

    testset = filter_test_to_train(testset, trainset)

    gt = defaultdict(set)
    for uid, iid, _ in testset:
        gt[str(uid)].add(str(iid))

    if len(gt) == 0:
        print(f"[WARN] Empty usable testset after filtering to trainset. model={model_name} variant={variant_name} seed={seed}")
        catalog_n = len(all_items_raw)
        return {
            "model": model_name,
            "variant": variant_name,
            "seed": int(seed),
            f"Precision@{topk}": float('nan'),
            f"Recall@{topk}": float('nan'),
            f"NDCG@{topk}": float('nan'),
            f"MRR@{topk}": float('nan'),
            f"Coverage@{topk}": float('nan'),
            "CatalogItems": float(catalog_n),
            f"UniqueRecItems@{topk}": float('nan'),
            f"ItemCV@{topk}": float('nan'),
            f"UserCoverage@{topk}": float('nan'),
            f"EEL@{topk}": float('nan'),
            "RecGini": float('nan'),
            "RecEntropy": float('nan'),
            f"1-Gini@{topk}": float('nan'),
            f"ARP@{topk}": float('nan'),
            f"APLT@{topk}": float('nan'),
            f"LTShare@{topk}": float('nan'),
            f"EPC@{topk}": float('nan'),
            f"EFD@{topk}": float('nan'),
            f"EPC_rel@{topk}": float('nan'),
            f"EFD_rel@{topk}": float('nan'),
            "ILS": float('nan'),
            "Diversity": float('nan'),
        }

    pop_counts = compute_popularity(trainset)
    long_tail = build_long_tail_set(pop_counts, pop_ratio=0.8)

    arp_vals, aplt_vals, lt_share_vals = [], [], []
    epc_vals, efd_vals = [], []
    epc_rel_vals, efd_rel_vals = [], []

    rec_items_global = set()
    rec_freq = Counter()

    recalls, ndcgs, precisions, mrrs = [], [], [], []
    ils_vals, div_vals = [], []
    user_hit_flags = []

    norm_topk = {str(k): [str(x) for x in v] for k, v in (user_topk or {}).items()}

    for uid in sorted(gt.keys()):
        ranked = norm_topk.get(str(uid), [])
        ranked = list(dict.fromkeys(ranked))[:topk]

        rec_items_global.update(ranked)
        rec_freq.update(ranked)

        precisions.append(precision_at_k(gt[uid], ranked, k=topk))
        recalls.append(recall_at_k(gt[uid], ranked, k=topk))
        ndcgs.append(ndcg_at_k(gt[uid], ranked, k=topk))
        mrrs.append(mrr_at_k(gt[uid], ranked, k=topk))
        user_hit_flags.append(int(any(i in gt[uid] for i in ranked)))

        arp_vals.append(arp_at_k(ranked, pop_counts, k=topk))
        aplt_vals.append(aplt_at_k(ranked, long_tail, pop_counts, trainset.n_users, k=topk, normalize=True))
        lt_share_vals.append(lt_share_at_k(ranked, long_tail, k=topk))
        epc_vals.append(epc_at_k(ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_vals.append(efd_at_k(ranked, pop_counts, k=topk, discounted=True))
        epc_rel_vals.append(epc_at_k_rel(gt[uid], ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_rel_vals.append(efd_at_k_rel(gt[uid], ranked, pop_counts, k=topk, discounted=True))

        if COMPUTE_ILS and item_users is not None:
            ils, div = ils_diversity(ranked, item_users)
            ils_vals.append(ils)
            div_vals.append(div)

    catalog_items = len(all_items_raw)
    unique_rec_items = len(rec_items_global)
    coverage = (unique_rec_items / catalog_items) if catalog_items else 0.0

    test_item_users = defaultdict(set)
    for u, items in gt.items():
        for iid in items:
            test_item_users[iid].add(u)
    U_test = len(gt)
    eel_terms = []
    all_items_test = set(test_item_users.keys()) | set(rec_freq.keys())
    for iid in all_items_test:
        rec_share = rec_freq.get(iid, 0) / max(U_test, 1)
        test_share = len(test_item_users.get(iid, set())) / max(U_test, 1)
        eel_terms.append(abs(rec_share - test_share))
    eel_value = float(np.mean(eel_terms)) if eel_terms else 0.0

    rec_counts_full = [rec_freq.get(iid, 0) for iid in all_items_raw]
    rec_gini = gini(rec_counts_full) if catalog_items else np.nan
    rec_entropy = shannon_entropy([c for c in rec_counts_full if c > 0])

    mean_rec = float(np.mean(rec_counts_full)) if rec_counts_full else 0.0
    std_rec = float(np.std(rec_counts_full)) if rec_counts_full else 0.0
    item_cv = float(std_rec / (mean_rec + 1e-12)) if mean_rec > 0 else 0.0

    out = {
    "model": model_name,
    "variant": variant_name,
    "seed": int(seed),

    f"Precision@{topk}": float(np.mean(precisions)) if precisions else 0.0,
    f"Recall@{topk}": float(np.mean(recalls)) if recalls else 0.0,
    f"NDCG@{topk}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    f"MRR@{topk}": float(np.mean(mrrs)) if mrrs else 0.0,

    f"Coverage@{topk}": float(coverage),
    "CatalogItems": int(catalog_items),
    f"UniqueRecItems@{topk}": int(unique_rec_items),
    f"ItemCV@{topk}": float(item_cv),
    f"UserCoverage@{topk}": float(np.mean(user_hit_flags)) if user_hit_flags else 0.0,
    }

    if COMPUTE_BEYOND_ACCURACY:
        out.update({
            f"EEL@{topk}": float(eel_value),
            "RecGini": float(rec_gini) if rec_gini == rec_gini else np.nan,
            "RecEntropy": float(rec_entropy) if rec_entropy == rec_entropy else np.nan,
            f"1-Gini@{topk}": float(1.0 - rec_gini) if rec_gini == rec_gini else np.nan,
            f"ARP@{topk}": float(np.mean(arp_vals)) if arp_vals else 0.0,
            f"APLT@{topk}": float(np.mean(aplt_vals)) if aplt_vals else 0.0,
            f"LTShare@{topk}": float(np.mean(lt_share_vals)) if lt_share_vals else 0.0,
            f"EPC@{topk}": float(np.mean(epc_vals)) if epc_vals else 0.0,
            f"EFD@{topk}": float(np.mean(efd_vals)) if efd_vals else 0.0,
            f"EPC_rel@{topk}": float(np.mean(epc_rel_vals)) if epc_rel_vals else 0.0,
            f"EFD_rel@{topk}": float(np.mean(efd_rel_vals)) if efd_rel_vals else 0.0,
        })
    else:
        out.update({
            f"EEL@{topk}": 0.0,
            "RecGini": 0.0,
            "RecEntropy": 0.0,
            f"1-Gini@{topk}": 0.0,
            f"ARP@{topk}": 0.0,
            f"APLT@{topk}": 0.0,
            f"LTShare@{topk}": 0.0,
            f"EPC@{topk}": 0.0,
            f"EFD@{topk}": 0.0,
            f"EPC_rel@{topk}": 0.0,
            f"EFD_rel@{topk}": 0.0,
        })

    out["ILS"] = float(np.mean(ils_vals)) if (COMPUTE_ILS and ils_vals) else 0.0
    out["Diversity"] = float(np.mean(div_vals)) if (COMPUTE_ILS and div_vals) else 0.0
    return out

def extra_report(frame: pd.DataFrame, topn: int = 10):
    print("\nTop users by activity:")
    print(frame["user_id"].value_counts().head(topn))
    print("\nTop items by popularity:")
    print(frame["item_id"].value_counts().head(topn))




raw_full_csv = str(VARIANTS_DIR / "ratings_raw_full.csv")
raw_aligned_csv = str(VARIANTS_DIR / "ratings_raw_aligned.csv")
sparse_csv = str(VARIANTS_DIR / "ratings_sparse.csv")
dense_csv = str(VARIANTS_DIR / "ratings_dense.csv")
raw_aligned_fixitems_csv = str(VARIANTS_DIR / "ratings_raw_aligned_fixitems.csv")
user_sparse_csv = str(VARIANTS_DIR / "ratings_user_sparse.csv")
user_dense_csv = str(VARIANTS_DIR / "ratings_user_dense.csv")










_evaluate_ranking_original = evaluate_ranking
_evaluate_ranking_from_saved_topk_original = evaluate_ranking_from_saved_topk


def _fixed_candidate_path_for_variant(variant_name: str, explicit_path=None) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    try:
        p = Path(common_split_paths(str(variant_name))["dir"]) / "fixed_candidates.json"
        return p if p.exists() else None
    except Exception:
        return None


def _normalise_fixed_candidates_payload(payload):


    if payload is None:
        return None, None

    if isinstance(payload, dict) and "users" in payload:
        raw_users = payload.get("users") or {}
        raw_universe = payload.get("candidate_universe", []) or []
    else:
        raw_users = payload or {}
        raw_universe = []

    users = {}
    universe = set(str(x) for x in raw_universe)

    for uid, spec in raw_users.items():
        uid_s = str(uid)
        if isinstance(spec, dict):
            positives = [str(x) for x in spec.get("positives", spec.get("positive", []))]
            candidates = [str(x) for x in spec.get("candidates", [])]
        else:
            positives = []
            candidates = [str(x) for x in (spec or [])]

        
        seen = set()
        cand_norm = []
        for x in list(candidates) + list(positives):
            x = str(x)
            if x not in seen:
                cand_norm.append(x)
                seen.add(x)

        pos_norm = []
        seen_pos = set()
        for x in positives:
            x = str(x)
            if x not in seen_pos:
                pos_norm.append(x)
                seen_pos.add(x)

        if not pos_norm:
            
            continue
        if not cand_norm:
            continue

        users[uid_s] = {"positives": pos_norm, "candidates": cand_norm}
        universe.update(cand_norm)

    return users, universe


def _load_fixed_candidates_for_variant(variant_name: str, explicit_path=None):
    p = _fixed_candidate_path_for_variant(variant_name, explicit_path=explicit_path)
    if p is None:
        return None, None, None
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and payload.get("ref_path"):
        ref = Path(str(payload["ref_path"]))
        if not ref.is_absolute():
            ref = (p.parent / ref).resolve()
        with open(ref, "r", encoding="utf-8") as f:
            payload = json.load(f)
        p = ref
    users, universe = _normalise_fixed_candidates_payload(payload)
    if not users:
        raise RuntimeError(f"[FIXED-EVAL] Empty fixed candidate file: {p}")
    return users, universe, p


def _fixed_eval_validate(users, trainset, testset, candidate_universe, strict=True):
    train_items = {str(trainset.to_raw_iid(i)) for i in trainset.all_items()}
    train_users = {str(trainset.to_raw_uid(u)) for u in trainset.all_users()}

    test_pairs = {(str(u), str(i)) for (u, i, _) in testset}

    missing_users = []
    missing_items = []
    missing_test_pairs = []

    for uid, spec in users.items():
        if uid not in train_users:
            missing_users.append(uid)
        for iid in spec["candidates"]:
            if iid not in train_items:
                missing_items.append(iid)
        for iid in spec["positives"]:
            if (uid, iid) not in test_pairs:
                missing_test_pairs.append((uid, iid))

    missing_items = sorted(set(missing_items))
    missing_users = sorted(set(missing_users))

    if strict and (missing_users or missing_items or missing_test_pairs):
        msg = ["[FIXED-EVAL] The fixed task is not scoreable for this variant."]
        if missing_users:
            msg.append(f"missing train users={len(missing_users)} example={missing_users[:5]}")
        if missing_items:
            msg.append(f"missing train candidate items={len(missing_items)} example={missing_items[:5]}")
        if missing_test_pairs:
            msg.append(f"fixed positives not found in testset={len(missing_test_pairs)} example={missing_test_pairs[:5]}")
        raise RuntimeError(" | ".join(msg))

    return train_users, train_items


def _evaluate_with_fixed_candidates(
    algo,
    trainset,
    testset,
    fixed_users,
    fixed_universe,
    fixed_path,
    topk=TOPK,
    save_ranked=False,
    model_name="model",
    variant_name="variant",
    save_path=EXPORT_DIR,
    seed=SEED,
    strict_fixed_eval=True,
):
    print(f"Ranking eval = FIXED CANDIDATES ({fixed_path}), TOPK={topk}")

    train_users, train_items = _fixed_eval_validate(
        fixed_users,
        trainset=trainset,
        testset=testset,
        candidate_universe=fixed_universe,
        strict=strict_fixed_eval,
    )

    user_hist = build_user_histories(trainset)
    item_users = build_item_user_sets(trainset) if COMPUTE_ILS else None
    pop_counts = compute_popularity(trainset)
    long_tail = build_long_tail_set(pop_counts, pop_ratio=0.8)

    
    eval_catalog = sorted(set(str(x) for x in fixed_universe))
    if not eval_catalog:
        eval_catalog = sorted({iid for spec in fixed_users.values() for iid in spec["candidates"]})

    arp_vals, aplt_vals, lt_share_vals = [], [], []
    epc_vals, efd_vals = [], []
    epc_rel_vals, efd_rel_vals = [], []
    rec_items_global = set()
    rec_freq = Counter()
    recalls, ndcgs, precisions, mrrs = [], [], [], []
    ils_vals, div_vals = [], []
    user_hit_flags = []
    user_topk = {}

    for uid in sorted(fixed_users.keys()):
        gt_items = set(fixed_users[uid]["positives"])
        candidates = list(dict.fromkeys(str(x) for x in fixed_users[uid]["candidates"]))

        
        for pos in gt_items:
            if pos not in candidates:
                candidates.append(pos)

        if hasattr(algo, "score_candidates"):
            scores = algo.score_candidates(uid, candidates)
            preds = list(zip(candidates, scores))
        else:
            preds = []
            err_count = 0
            max_err_print = 10
            for iid in candidates:
                try:
                    est = algo.predict(uid, iid, verbose=False).est
                except Exception as e:
                    if err_count < max_err_print:
                        import traceback
                        print(f"[PRED-ERR] model={model_name} variant={variant_name} uid={uid} iid={iid} err={repr(e)}")
                        traceback.print_exc()
                    err_count += 1
                    est = 0.0
                preds.append((iid, est))
            if err_count > 0:
                print(f"[PRED-ERR-SUMMARY] model={model_name} variant={variant_name} total_errors={err_count}")

        
        
        preds.sort(key=lambda x: x[1], reverse=True)
        ranked = [iid for iid, _ in preds[:topk]]

        rec_freq.update(ranked)
        rec_items_global.update(ranked)

        recalls.append(recall_at_k(gt_items, ranked, k=topk))
        ndcgs.append(ndcg_at_k(gt_items, ranked, k=topk))
        precisions.append(precision_at_k(gt_items, ranked, k=topk))
        mrrs.append(mrr_at_k(gt_items, ranked, k=topk))
        user_hit_flags.append(int(any(i in gt_items for i in ranked)))

        arp_vals.append(arp_at_k(ranked, pop_counts, k=topk))
        aplt_vals.append(aplt_at_k(ranked, long_tail, pop_counts, trainset.n_users, k=topk, normalize=True))
        lt_share_vals.append(lt_share_at_k(ranked, long_tail, k=topk))
        epc_vals.append(epc_at_k(ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_vals.append(efd_at_k(ranked, pop_counts, k=topk, discounted=True))
        epc_rel_vals.append(epc_at_k_rel(gt_items, ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_rel_vals.append(efd_at_k_rel(gt_items, ranked, pop_counts, k=topk, discounted=True))

        if COMPUTE_ILS and item_users is not None:
            ils, div = ils_diversity(ranked, item_users)
            ils_vals.append(ils)
            div_vals.append(div)

        if save_ranked:
            user_topk[str(uid)] = ranked

    if save_ranked:
        try:
            out_file = Path(save_path) / f"topk_{variant_name}_{model_name}_seed{seed}.json"
            _write_json_no_overwrite({str(k): v for k, v in user_topk.items()}, out_file)
        except Exception as e:
            print(f"[WARN] Could not save Top-K JSON: {repr(e)}")

    catalog_items = len(eval_catalog)
    unique_rec_items = len(rec_items_global)
    coverage = (unique_rec_items / catalog_items) if catalog_items else 0.0

    
    test_item_users = defaultdict(set)
    for uid, spec in fixed_users.items():
        for iid in spec["positives"]:
            test_item_users[iid].add(uid)
    U_test = len(fixed_users)
    eel_terms = []
    all_items_test = set(test_item_users.keys()) | set(rec_freq.keys())
    for iid in all_items_test:
        rec_share = rec_freq.get(iid, 0) / max(U_test, 1)
        test_share = len(test_item_users.get(iid, set())) / max(U_test, 1)
        eel_terms.append(abs(rec_share - test_share))
    eel_value = float(np.mean(eel_terms)) if eel_terms else 0.0

    rec_counts_full = [rec_freq.get(iid, 0) for iid in eval_catalog]
    rec_gini = gini(rec_counts_full) if catalog_items else np.nan
    rec_entropy = shannon_entropy([c for c in rec_counts_full if c > 0])
    mean_rec = float(np.mean(rec_counts_full)) if rec_counts_full else 0.0
    std_rec = float(np.std(rec_counts_full)) if rec_counts_full else 0.0
    item_cv = float(std_rec / (mean_rec + 1e-12)) if mean_rec > 0 else 0.0

    out = {
        f"Precision@{topk}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{topk}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{topk}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MRR@{topk}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"Coverage@{topk}": float(coverage),
        "CatalogItems": int(catalog_items),
        f"UniqueRecItems@{topk}": int(unique_rec_items),
        f"ItemCV@{topk}": float(item_cv),
        f"UserCoverage@{topk}": float(np.mean(user_hit_flags)) if user_hit_flags else 0.0,
        "FixedEval": 1,
        "FixedEvalUsers": int(len(fixed_users)),
        "FixedEvalCandidates": int(sum(len(v["candidates"]) for v in fixed_users.values())),
    }

    if COMPUTE_BEYOND_ACCURACY:
        out.update({
            f"EEL@{topk}": float(eel_value),
            "RecGini": float(rec_gini) if rec_gini == rec_gini else np.nan,
            "RecEntropy": float(rec_entropy) if rec_entropy == rec_entropy else np.nan,
            f"1-Gini@{topk}": float(1.0 - rec_gini) if rec_gini == rec_gini else np.nan,
            f"ARP@{topk}": float(np.mean(arp_vals)) if arp_vals else 0.0,
            f"APLT@{topk}": float(np.mean(aplt_vals)) if aplt_vals else 0.0,
            f"LTShare@{topk}": float(np.mean(lt_share_vals)) if lt_share_vals else 0.0,
            f"EPC@{topk}": float(np.mean(epc_vals)) if epc_vals else 0.0,
            f"EFD@{topk}": float(np.mean(efd_vals)) if efd_vals else 0.0,
            f"EPC_rel@{topk}": float(np.mean(epc_rel_vals)) if epc_rel_vals else 0.0,
            f"EFD_rel@{topk}": float(np.mean(efd_rel_vals)) if efd_rel_vals else 0.0,
        })
    else:
        out.update({
            f"EEL@{topk}": 0.0,
            "RecGini": 0.0,
            "RecEntropy": 0.0,
            f"1-Gini@{topk}": 0.0,
            f"ARP@{topk}": 0.0,
            f"APLT@{topk}": 0.0,
            f"LTShare@{topk}": 0.0,
            f"EPC@{topk}": 0.0,
            f"EFD@{topk}": 0.0,
            f"EPC_rel@{topk}": 0.0,
            f"EFD_rel@{topk}": 0.0,
        })

    if COMPUTE_ILS and ils_vals:
        out["ILS"] = float(np.mean(ils_vals))
        out["Diversity"] = float(np.mean(div_vals)) if div_vals else 0.0
    else:
        out["ILS"] = 0.0
        out["Diversity"] = 0.0

    return out


def evaluate_ranking(
    algo,
    trainset,
    testset,
    topk=TOPK,
    save_ranked=False,
    model_name="model",
    variant_name="variant",
    save_path=EXPORT_DIR,
    neg_sample=NEGATIVE_CANDIDATE_SAMPLE,
    seed=SEED,
    fixed_candidates_path=None,
    strict_fixed_eval=True,
):
    fixed_users, fixed_universe, fixed_path = _load_fixed_candidates_for_variant(
        variant_name,
        explicit_path=fixed_candidates_path,
    )
    if fixed_users is None:
        return _evaluate_ranking_original(
            algo=algo,
            trainset=trainset,
            testset=testset,
            topk=topk,
            save_ranked=save_ranked,
            model_name=model_name,
            variant_name=variant_name,
            save_path=save_path,
            neg_sample=neg_sample,
            seed=seed,
        )

    return _evaluate_with_fixed_candidates(
        algo=algo,
        trainset=trainset,
        testset=testset,
        fixed_users=fixed_users,
        fixed_universe=fixed_universe,
        fixed_path=fixed_path,
        topk=topk,
        save_ranked=save_ranked,
        model_name=model_name,
        variant_name=variant_name,
        save_path=save_path,
        seed=seed,
        strict_fixed_eval=strict_fixed_eval,
    )


def evaluate_ranking_from_saved_topk(
    user_topk: dict,
    trainset,
    testset,
    topk=TOPK,
    model_name="model",
    variant_name="variant",
    seed=SEED,
    fixed_candidates_path=None,
    strict_fixed_eval=True,
):
    fixed_users, fixed_universe, fixed_path = _load_fixed_candidates_for_variant(
        variant_name,
        explicit_path=fixed_candidates_path,
    )
    if fixed_users is None:
        return _evaluate_ranking_from_saved_topk_original(
            user_topk=user_topk,
            trainset=trainset,
            testset=testset,
            topk=topk,
            model_name=model_name,
            variant_name=variant_name,
            seed=seed,
        )

    print(f"Offline ranking eval = FIXED CANDIDATES ({fixed_path}), TOPK={topk}")
    _fixed_eval_validate(
        fixed_users,
        trainset=trainset,
        testset=testset,
        candidate_universe=fixed_universe,
        strict=strict_fixed_eval,
    )

    norm_topk = {str(k): [str(x) for x in v] for k, v in (user_topk or {}).items()}
    eval_catalog = sorted(set(str(x) for x in fixed_universe))
    if not eval_catalog:
        eval_catalog = sorted({iid for spec in fixed_users.values() for iid in spec["candidates"]})

    item_users = build_item_user_sets(trainset) if COMPUTE_ILS else None
    pop_counts = compute_popularity(trainset)
    long_tail = build_long_tail_set(pop_counts, pop_ratio=0.8)

    arp_vals, aplt_vals, lt_share_vals = [], [], []
    epc_vals, efd_vals = [], []
    epc_rel_vals, efd_rel_vals = [], []
    rec_items_global = set()
    rec_freq = Counter()
    recalls, ndcgs, precisions, mrrs = [], [], [], []
    ils_vals, div_vals = [], []
    user_hit_flags = []

    for uid in sorted(fixed_users.keys()):
        gt_items = set(fixed_users[uid]["positives"])
        ranked = norm_topk.get(str(uid), [])
        ranked = list(dict.fromkeys(ranked))[:topk]

        rec_items_global.update(ranked)
        rec_freq.update(ranked)

        precisions.append(precision_at_k(gt_items, ranked, k=topk))
        recalls.append(recall_at_k(gt_items, ranked, k=topk))
        ndcgs.append(ndcg_at_k(gt_items, ranked, k=topk))
        mrrs.append(mrr_at_k(gt_items, ranked, k=topk))
        user_hit_flags.append(int(any(i in gt_items for i in ranked)))

        arp_vals.append(arp_at_k(ranked, pop_counts, k=topk))
        aplt_vals.append(aplt_at_k(ranked, long_tail, pop_counts, trainset.n_users, k=topk, normalize=True))
        lt_share_vals.append(lt_share_at_k(ranked, long_tail, k=topk))
        epc_vals.append(epc_at_k(ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_vals.append(efd_at_k(ranked, pop_counts, k=topk, discounted=True))
        epc_rel_vals.append(epc_at_k_rel(gt_items, ranked, pop_counts, trainset.n_users, k=topk, discounted=True))
        efd_rel_vals.append(efd_at_k_rel(gt_items, ranked, pop_counts, k=topk, discounted=True))

        if COMPUTE_ILS and item_users is not None:
            ils, div = ils_diversity(ranked, item_users)
            ils_vals.append(ils)
            div_vals.append(div)

    catalog_items = len(eval_catalog)
    unique_rec_items = len(rec_items_global)
    coverage = (unique_rec_items / catalog_items) if catalog_items else 0.0

    test_item_users = defaultdict(set)
    for uid, spec in fixed_users.items():
        for iid in spec["positives"]:
            test_item_users[iid].add(uid)
    U_test = len(fixed_users)
    eel_terms = []
    all_items_test = set(test_item_users.keys()) | set(rec_freq.keys())
    for iid in all_items_test:
        rec_share = rec_freq.get(iid, 0) / max(U_test, 1)
        test_share = len(test_item_users.get(iid, set())) / max(U_test, 1)
        eel_terms.append(abs(rec_share - test_share))
    eel_value = float(np.mean(eel_terms)) if eel_terms else 0.0

    rec_counts_full = [rec_freq.get(iid, 0) for iid in eval_catalog]
    rec_gini = gini(rec_counts_full) if catalog_items else np.nan
    rec_entropy = shannon_entropy([c for c in rec_counts_full if c > 0])
    mean_rec = float(np.mean(rec_counts_full)) if rec_counts_full else 0.0
    std_rec = float(np.std(rec_counts_full)) if rec_counts_full else 0.0
    item_cv = float(std_rec / (mean_rec + 1e-12)) if mean_rec > 0 else 0.0

    out = {
        "model": model_name,
        "variant": variant_name,
        "seed": int(seed),
        f"Precision@{topk}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{topk}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{topk}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MRR@{topk}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"Coverage@{topk}": float(coverage),
        "CatalogItems": int(catalog_items),
        f"UniqueRecItems@{topk}": int(unique_rec_items),
        f"ItemCV@{topk}": float(item_cv),
        f"UserCoverage@{topk}": float(np.mean(user_hit_flags)) if user_hit_flags else 0.0,
        "FixedEval": 1,
        "FixedEvalUsers": int(len(fixed_users)),
        "FixedEvalCandidates": int(sum(len(v["candidates"]) for v in fixed_users.values())),
    }

    if COMPUTE_BEYOND_ACCURACY:
        out.update({
            f"EEL@{topk}": float(eel_value),
            "RecGini": float(rec_gini) if rec_gini == rec_gini else np.nan,
            "RecEntropy": float(rec_entropy) if rec_entropy == rec_entropy else np.nan,
            f"1-Gini@{topk}": float(1.0 - rec_gini) if rec_gini == rec_gini else np.nan,
            f"ARP@{topk}": float(np.mean(arp_vals)) if arp_vals else 0.0,
            f"APLT@{topk}": float(np.mean(aplt_vals)) if aplt_vals else 0.0,
            f"LTShare@{topk}": float(np.mean(lt_share_vals)) if lt_share_vals else 0.0,
            f"EPC@{topk}": float(np.mean(epc_vals)) if epc_vals else 0.0,
            f"EFD@{topk}": float(np.mean(efd_vals)) if efd_vals else 0.0,
            f"EPC_rel@{topk}": float(np.mean(epc_rel_vals)) if epc_rel_vals else 0.0,
            f"EFD_rel@{topk}": float(np.mean(efd_rel_vals)) if efd_rel_vals else 0.0,
        })
    else:
        out.update({
            f"EEL@{topk}": 0.0,
            "RecGini": 0.0,
            "RecEntropy": 0.0,
            f"1-Gini@{topk}": 0.0,
            f"ARP@{topk}": 0.0,
            f"APLT@{topk}": 0.0,
            f"LTShare@{topk}": 0.0,
            f"EPC@{topk}": 0.0,
            f"EFD@{topk}": 0.0,
            f"EPC_rel@{topk}": 0.0,
            f"EFD_rel@{topk}": 0.0,
        })

    out["ILS"] = float(np.mean(ils_vals)) if (COMPUTE_ILS and ils_vals) else 0.0
    out["Diversity"] = float(np.mean(div_vals)) if (COMPUTE_ILS and div_vals) else 0.0
    return out
