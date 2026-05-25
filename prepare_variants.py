
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    BINARIZE_FROM_EXPLICIT,
    DATASET_NAME,
    EXPORT_DIR,
    IMPLICIT_MIN_POS_PER_USER,
    INPUT_RATINGS_PATH,
    META_DIR,
    RUN_PREPARE_VARIANTS,
    RUN_ROOT,
    RUN_SPARSITY_REPORTS,
    SEED,
    VARIANTS_DIR,
    dense_csv,
    drop_items_by_pop_percentile,
    filter_by_activity_iterative,
    maybe_binarize,
    normalize_cols_any,
    quick_stats,
    raw_aligned_csv,
    raw_aligned_fixitems_csv,
    raw_full_csv,
    read_csv_smart,
    read_interactions_csv,
    sample_items_aligned,
    sample_users_aligned,
    sample_users_fixed_size,
    save_variant,
    sparse_csv,
    sparsity_report,
    user_dense_csv,
    user_sparse_csv,
)

COMMON_SPLITS_DIR = RUN_ROOT / "splits"
COMMON_SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def _variant_name_from_csv_path(path_like) -> str:
    stem = Path(path_like).stem
    return stem[len("ratings_"):] if stem.startswith("ratings_") else stem


def _common_split_paths(variant_name: str):
    split_dir = COMMON_SPLITS_DIR / variant_name
    return {
        "dir": split_dir,
        "train_csv": split_dir / "train.csv",
        "test_csv": split_dir / "test.csv",
        "meta_json": split_dir / "meta.json",
    }

def _interaction_keep_cols(df: pd.DataFrame):
    return [c for c in ["user_id", "item_id", "rating", "timestamp"] if c in df.columns]


def _deduplicate_latest(df: pd.DataFrame, time_col: str = "timestamp"):
    work = df[_interaction_keep_cols(df)].copy()
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


def _split_user_loo(df: pd.DataFrame, seed: int = SEED, shuffle: bool = True):


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
            raise RuntimeError("Chronological split was requested, but no timestamp column is available.")

        work = work.dropna(subset=[time_col]).copy()
        work = _deduplicate_latest(work, time_col=time_col)
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

    work = _deduplicate_latest(work)
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


def _save_common_split_for_variant_csv(csv_path: str, seed: int = SEED, shuffle: bool = True, force: bool = False):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    variant_name = _variant_name_from_csv_path(csv_path)
    paths = _common_split_paths(variant_name)
    split_dir = paths["dir"]
    split_dir.mkdir(parents=True, exist_ok=True)

    if (not force) and paths["train_csv"].exists() and paths["test_csv"].exists() and paths["meta_json"].exists():
        print(f"[SPLIT][SKIP] Already exists: {variant_name}")
        return paths

    dfv = read_csv_smart(csv_path)
    dfv = normalize_cols_any(dfv)
    dfv = dfv[_interaction_keep_cols(dfv)].copy()

    train_df, test_df = _split_user_loo(dfv, seed=seed, shuffle=shuffle)

    train_df.to_csv(paths["train_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)

    effective_split_type = (
        "chrono_loo"
        if ("timestamp" in dfv.columns and dfv["timestamp"].notna().any()
            and os.getenv("COMMON_SPLIT_MODE", "auto").strip().lower() != "random_loo")
        else "random_loo"
    )

    meta = {
        "variant": variant_name,
        "source_csv": str(csv_path),
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
    }
    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[SPLIT] {variant_name}: train_rows={len(train_df):,} | test_rows={len(test_df):,} | "
        f"train_users={train_df['user_id'].nunique():,} | test_users={test_df['user_id'].nunique():,}"
    )
    print(f"-> Saved split: {split_dir}")
    return paths


def _build_common_splits_for_all_variant_csvs(force: bool = False):
    split_shuffle = _env_bool("COMMON_SPLIT_SHUFFLE", False)
    split_seed = _env_int("COMMON_SPLIT_SEED", SEED)
    split_glob = os.getenv("COMMON_SPLIT_FILE_GLOB", "ratings_*.csv").strip() or "ratings_*.csv"

    csv_paths = _csv_paths_from_globs(Path(VARIANTS_DIR), split_glob)
    if not csv_paths:
        print(f"[SPLIT] No variant CSV files found for glob: {split_glob}")
        return

    print(f"[SPLIT] Building common LOO splits for {len(csv_paths)} variant file(s), glob={split_glob!r}...")
    for csv_path in csv_paths:
        _save_common_split_for_variant_csv(
            csv_path=str(csv_path),
            seed=split_seed,
            shuffle=split_shuffle,
            force=force,
        )

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key, "").strip()
    if not v:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key, "").strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_float_list(s: str, default_list):
    if s is None:
        return list(default_list)
    out = []
    for part in str(s).replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    out = [x for x in out if x > 0]
    return out if out else list(default_list)


def _parse_int_list(s: str, default_list):
    if s is None:
        return [int(x) for x in default_list]
    out = []
    for part in str(s).replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    out = [x for x in out if x > 0]
    return out if out else [int(x) for x in default_list]


def _csv_paths_from_globs(base_dir: Path, glob_spec: str):
    patterns = []
    for part in str(glob_spec or "").replace(";", ",").split(","):
        pat = part.strip()
        if pat:
            patterns.append(pat)
    if not patterns:
        patterns = ["ratings_*.csv"]

    found = []
    seen = set()
    for pat in patterns:
        for pth in sorted(Path(base_dir).glob(pat)):
            key = str(pth.resolve())
            if key not in seen:
                found.append(pth)
                seen.add(key)
    return found


def _pct_label(p: float) -> str:
    val = float(p) * 100.0
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    s = f"{val:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _addback_variant_name(target_k: int, source_k: int, pct: float, seed: int) -> str:
    return f"kcore{int(target_k)}_addback_from_kcore{int(source_k)}_p{_pct_label(pct)}_seed{int(seed)}"


def _taildrop_variant_name(pct: float) -> str:
    return f"td{_pct_label(float(pct))}"


def _parse_randdrop_base(spec: str):
    spec = (spec or "raw").strip().lower()
    if spec.startswith("kcore"):
        tail = spec.replace("kcore", "").strip(":_-")
        try:
            return ("kcore", int(tail))
        except Exception:
            return ("kcore", 10)
    return ("raw", None)


def _pick_fixed_sets(df: pd.DataFrame, target_users: int, target_items: int, seed: int, min_user: int, min_item: int):


    base0 = _deduplicate_latest(normalize_cols_any(df))

    
    base0 = filter_by_activity_iterative(base0, min_user=min_user, min_item=min_item)

    
    if target_users and target_users > 0:
        base0 = sample_users_fixed_size(base0, target_users=int(target_users), seed=seed)

    
    if target_items and target_items > 0:
        items = base0["item_id"].unique()
        if len(items) > int(target_items):
            rng = np.random.default_rng(seed + 1)
            keep_items = set(rng.choice(items, size=int(target_items), replace=False))
            base0 = base0[base0["item_id"].isin(keep_items)].copy()

    
    base0 = filter_by_activity_iterative(base0, min_user=min_user, min_item=min_item)

    base0 = base0.sort_values(["user_id", "item_id"]).drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)
    return base0


def _sample_to_user_item_cardinality(
    df: pd.DataFrame,
    target_users: int,
    target_items: int,
    seed: int,
    max_attempts: int = 200,
    forbidden_user_set=None,
    forbidden_item_set=None,
):
    """
    Random size-matched k-core control.

    This matches only the number of retained users and items. It must not lock
    the matched source to the target k-core user IDs. If forbidden_user_set is
    supplied, samples whose final user set is identical to that set are rejected.
    Interaction count, density, degree distribution, popularity profile, and
    user/item identities are otherwise intentionally left uncontrolled.
    """
    base = _deduplicate_latest(normalize_cols_any(df))
    base = base.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

    target_users = int(target_users)
    target_items = int(target_items)

    if target_users <= 0 or target_items <= 0:
        raise RuntimeError("[KCORE SIZE MATCH] Target user/item size must be positive.")

    n_users = int(base["user_id"].nunique())
    n_items = int(base["item_id"].nunique())

    if n_users < target_users:
        raise RuntimeError(
            f"[KCORE SIZE MATCH] Source has too few users: source_users={n_users}, target_users={target_users}"
        )
    if n_items < target_items:
        raise RuntimeError(
            f"[KCORE SIZE MATCH] Source has too few items: source_items={n_items}, target_items={target_items}"
        )

    forbidden_user_set = set(forbidden_user_set) if forbidden_user_set is not None else None
    forbidden_item_set = set(forbidden_item_set) if forbidden_item_set is not None else None

    all_users = base["user_id"].drop_duplicates().to_numpy()

    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(int(seed) + attempt * 7919)

        chosen_users_seed = set(rng.choice(all_users, size=target_users, replace=False).tolist())
        sub_u = base[base["user_id"].isin(chosen_users_seed)].copy()

        candidate_items = sub_u["item_id"].drop_duplicates().to_numpy()
        if len(candidate_items) < target_items:
            continue

        chosen_items = set(rng.choice(candidate_items, size=target_items, replace=False).tolist())
        sub = sub_u[sub_u["item_id"].isin(chosen_items)].copy()

        present_users = set(sub["user_id"].drop_duplicates().tolist())
        if len(present_users) < target_users:
            missing = target_users - len(present_users)
            eligible_extra = (
                base.loc[
                    base["item_id"].isin(chosen_items)
                    & (~base["user_id"].isin(present_users)),
                    "user_id",
                ]
                .drop_duplicates()
                .to_numpy()
            )
            if len(eligible_extra) < missing:
                continue
            add_users = set(rng.choice(eligible_extra, size=missing, replace=False).tolist())
            chosen_users = present_users | add_users
        elif len(present_users) > target_users:
            chosen_users = set(rng.choice(np.asarray(list(present_users), dtype=object), size=target_users, replace=False).tolist())
        else:
            chosen_users = present_users

        out = base[base["user_id"].isin(chosen_users) & base["item_id"].isin(chosen_items)].copy()
        out = out.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

        final_user_set = set(out["user_id"].drop_duplicates().tolist())
        final_item_set = set(out["item_id"].drop_duplicates().tolist())
        final_users = int(len(final_user_set))
        final_items = int(len(final_item_set))

        if final_users != target_users or final_items != target_items or len(out) == 0:
            continue

        same_user_set_as_target = bool(forbidden_user_set is not None and final_user_set == forbidden_user_set)
        same_item_set_as_target = bool(forbidden_item_set is not None and final_item_set == forbidden_item_set)

        if same_user_set_as_target:
            continue

        out = out.sort_values(["user_id", "item_id"], kind="mergesort").reset_index(drop=True)
        meta = {
            "seed": int(seed),
            "attempt": int(attempt),
            "target_users": int(target_users),
            "target_items": int(target_items),
            "source_users": int(n_users),
            "source_items": int(n_items),
            "matched_users": int(final_users),
            "matched_items": int(final_items),
            "matched_rows": int(len(out)),
            "same_user_set_as_target": bool(same_user_set_as_target),
            "same_item_set_as_target": bool(same_item_set_as_target),
            "target_user_overlap_count": int(len(final_user_set & forbidden_user_set)) if forbidden_user_set is not None else None,
            "target_user_overlap_ratio": (
                float(len(final_user_set & forbidden_user_set) / target_users) if forbidden_user_set is not None and target_users else None
            ),
            "target_item_overlap_count": int(len(final_item_set & forbidden_item_set)) if forbidden_item_set is not None else None,
            "target_item_overlap_ratio": (
                float(len(final_item_set & forbidden_item_set) / target_items) if forbidden_item_set is not None and target_items else None
            ),
            "identity_policy": "match user/item cardinality only; reject exact target user set",
        }
        return out, meta

    raise RuntimeError(
        f"[KCORE SIZE MATCH] Could not build an exact non-target-user-set size match in {max_attempts} attempts. "
        f"target_users={target_users}, target_items={target_items}, source_users={n_users}, source_items={n_items}"
    )


def _write_size_match_meta(name: str, meta: dict):
    out_dir = Path(META_DIR) / "kcore_size_match"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return out_path


def _assert_fixed_sets(base_fixed: pd.DataFrame, kept: pd.DataFrame, tag: str):
    bu = set(base_fixed["user_id"].unique())
    bi = set(base_fixed["item_id"].unique())
    ku = set(kept["user_id"].unique())
    ki = set(kept["item_id"].unique())

    if bu != ku:
        raise RuntimeError(f"[RANDDROP][{tag}] User set is not fixed: base={len(bu)} kept={len(ku)}")
    if bi != ki:
        raise RuntimeError(f"[RANDDROP][{tag}] Item set is not fixed: base={len(bi)} kept={len(ki)}")

    base_pairs = set(zip(base_fixed["user_id"], base_fixed["item_id"]))
    kept_pairs = set(zip(kept["user_id"], kept["item_id"]))
    if not kept_pairs.issubset(base_pairs):
        raise RuntimeError(f"[RANDDROP][{tag}] Kept graph contains a user-item pair that is not present in base_fixed.")



def _safe_random_drop(base_fixed: pd.DataFrame, rng: np.random.Generator,
                      min_user: int, min_item: int, n_drop: int):


    df = _deduplicate_latest(normalize_cols_any(base_fixed))

    udeg = df["user_id"].value_counts().to_dict()
    ideg = df["item_id"].value_counts().to_dict()

    idxs = np.arange(len(df))
    rng.shuffle(idxs)

    drop_mask = np.zeros(len(df), dtype=bool)
    dropped = 0

    for ix in idxs:
        if dropped >= n_drop:
            break
        u = df.at[ix, "user_id"]
        i = df.at[ix, "item_id"]

        if (min_user > 0 and udeg.get(u, 0) <= min_user):
            continue
        if (min_item > 0 and ideg.get(i, 0) <= min_item):
            continue

        drop_mask[ix] = True
        dropped += 1
        udeg[u] = udeg.get(u, 0) - 1
        ideg[i] = ideg.get(i, 0) - 1

    kept = df.loc[~drop_mask].copy().reset_index(drop=True)
    return kept, int(dropped)


def _rescue_min_degree(base_fixed: pd.DataFrame, kept: pd.DataFrame, rng: np.random.Generator,
                       min_user: int, min_item: int, max_rounds: int = 10):
    out = _deduplicate_latest(normalize_cols_any(kept))

    
    target_users = set(base_fixed["user_id"].unique())
    target_items = set(base_fixed["item_id"].unique())

    
    user_to_idx = {}
    item_to_idx = {}
    for ix, row in base_fixed[["user_id", "item_id"]].iterrows():
        u = row["user_id"]
        i = row["item_id"]
        user_to_idx.setdefault(u, []).append(ix)
        item_to_idx.setdefault(i, []).append(ix)

    base_pairs = set(zip(base_fixed["user_id"], base_fixed["item_id"]))
    out_pairs = set(zip(out["user_id"], out["item_id"]))

    for _ in range(max_rounds):
        ucnt = out.groupby("user_id").size()
        icnt = out.groupby("item_id").size()

        present_u = set(out["user_id"].unique())
        present_i = set(out["item_id"].unique())

        
        missing_users = list(target_users - present_u)
        missing_items = list(target_items - present_i)

        
        low_users = [u for u, c in ucnt.items() if c < min_user] if min_user > 0 else []
        low_items = [i for i, c in icnt.items() if c < min_item] if min_item > 0 else []

        need_users = list(dict.fromkeys(missing_users + low_users))
        need_items = list(dict.fromkeys(missing_items + low_items))

        if not need_users and not need_items:
            break

        add_rows = []

        for u in need_users:
            current = int(ucnt.get(u, 0))
            need = max(0, min_user - current)
            if need == 0:
                continue
            idxs = user_to_idx.get(u, [])
            cand = [ix for ix in idxs
                    if (base_fixed.loc[ix, "user_id"], base_fixed.loc[ix, "item_id"]) in base_pairs
                    and (base_fixed.loc[ix, "user_id"], base_fixed.loc[ix, "item_id"]) not in out_pairs]
            if cand:
                take = min(need, len(cand))
                chosen = rng.choice(np.array(cand, dtype=int), size=take, replace=False)
                add_rows.append(base_fixed.loc[chosen, _interaction_keep_cols(base_fixed)])

        for it in need_items:
            current = int(icnt.get(it, 0))
            need = max(0, min_item - current)
            if need == 0:
                continue
            idxs = item_to_idx.get(it, [])
            cand = [ix for ix in idxs
                    if (base_fixed.loc[ix, "user_id"], base_fixed.loc[ix, "item_id"]) in base_pairs
                    and (base_fixed.loc[ix, "user_id"], base_fixed.loc[ix, "item_id"]) not in out_pairs]
            if cand:
                take = min(need, len(cand))
                chosen = rng.choice(np.array(cand, dtype=int), size=take, replace=False)
                add_rows.append(base_fixed.loc[chosen, _interaction_keep_cols(base_fixed)])

        if add_rows:
            out = pd.concat([out] + add_rows, ignore_index=True)
            out = out.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)
            out_pairs = set(zip(out["user_id"], out["item_id"]))
        else:
            break

    
    ucnt2 = out.groupby("user_id").size()
    icnt2 = out.groupby("item_id").size()

    min_u = int(ucnt2.reindex(list(target_users), fill_value=0).min()) if target_users else 0
    min_i = int(icnt2.reindex(list(target_items), fill_value=0).min()) if target_items else 0

    if min_user > 0 and min_u < min_user:
        raise RuntimeError(f"[RANDDROP] Rescue failed: at least one user is below min_user: min_u={min_u} < {min_user}")
    if min_item > 0 and min_i < min_item:
        raise RuntimeError(f"[RANDDROP] Rescue failed: at least one item is below min_item: min_i={min_i} < {min_item}")


    return out


def _build_common_splits_for_selected_variant_csvs(csv_paths, seed: int = None, shuffle: bool = None, force: bool = False):
    if seed is None:
        seed = _env_int("COMMON_SPLIT_SEED", SEED)
    if shuffle is None:
        shuffle = _env_bool("COMMON_SPLIT_SHUFFLE", False)

    csv_paths = [Path(p) for p in csv_paths]
    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        print("[SPLIT] No selected variant CSV files found.")
        return []

    out = []
    print(f"[SPLIT] Building common LOO splits for {len(csv_paths)} selected variant file(s)...")
    for csv_path in csv_paths:
        out.append(
            _save_common_split_for_variant_csv(
                csv_path=str(csv_path),
                seed=seed,
                shuffle=shuffle,
                force=force,
            )
        )
    return out



def _remove_user_item_pairs(df: pd.DataFrame, pairs_df: pd.DataFrame) -> pd.DataFrame:

    work = _deduplicate_latest(normalize_cols_any(df))
    if pairs_df is None or len(pairs_df) == 0:
        return work.reset_index(drop=True)

    keys = normalize_cols_any(pairs_df)[["user_id", "item_id"]].drop_duplicates().copy()
    keys["__drop"] = 1
    out = work.merge(keys, on=["user_id", "item_id"], how="left")
    out = out[out["__drop"].isna()].drop(columns=["__drop"]).reset_index(drop=True)
    return out


def _write_fixed_candidates_reference(variant_name: str, fixed_candidates_path: Path):

    paths = _common_split_paths(variant_name)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    ref_path = paths["dir"] / "fixed_candidates.json"
    payload = {
        "format": "fixed_candidates_ref_v1",
        "ref_path": str(Path(fixed_candidates_path).resolve()),
    }
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return ref_path


def _load_or_build_kcore_plain_for_addback(k: int) -> pd.DataFrame:


    k = int(k)
    existing = Path(VARIANTS_DIR) / f"ratings_kcore{k}_plain.csv"
    if existing.exists():
        return _deduplicate_latest(normalize_cols_any(read_csv_smart(existing)))

    print(f"[ADDBACK COMMON TEST] ratings_kcore{k}_plain.csv not found; rebuilding from input.")
    base = read_interactions_csv(INPUT_RATINGS_PATH)
    base_imp, used_thr = maybe_binarize(
        base,
        enabled=BINARIZE_FROM_EXPLICIT,
        ensure_min_pos_user=IMPLICIT_MIN_POS_PER_USER,
    )

    
    lastfm_item_cap = _env_int("LASTFM_GLOBAL_ITEM_CAP", 0)
    if DATASET_NAME.lower() == "lastfm" and lastfm_item_cap > 0:
        uniq_items = base_imp["item_id"].drop_duplicates().to_numpy()
        if len(uniq_items) > lastfm_item_cap:
            rng = np.random.default_rng(SEED)
            keep_items = set(rng.choice(uniq_items, size=lastfm_item_cap, replace=False))
            base_imp = base_imp[base_imp["item_id"].isin(keep_items)].copy().reset_index(drop=True)

    out = filter_by_activity_iterative(base_imp, min_user=k, min_item=k)
    quick_stats(out, f"rebuilt_kcore{k}_plain_for_common_test")
    return _deduplicate_latest(normalize_cols_any(out))


def _make_fixed_target_test_split(
    target_df: pd.DataFrame,
    seed: int,
    shuffle: bool,
):


    target_df = _deduplicate_latest(normalize_cols_any(target_df))
    train0, test0 = _split_user_loo(target_df, seed=seed, shuffle=shuffle)

    train_users0 = set(train0["user_id"].drop_duplicates().tolist())
    train_items0 = set(train0["item_id"].drop_duplicates().tolist())

    fixed_test = test0[
        test0["user_id"].isin(train_users0)
        & test0["item_id"].isin(train_items0)
    ].copy().reset_index(drop=True)

    dropped = int(len(test0) - len(fixed_test))
    if dropped > 0:
        print(
            f"[ADDBACK COMMON TEST][WARN] Dropped {dropped:,} target test pair(s) "
            "because the user or item would be cold-start in the anchor train graph. "
            "These pairs are returned to train and are not evaluated."
        )

    if fixed_test.empty:
        raise RuntimeError("[ADDBACK COMMON TEST] Fixed target test is empty after scoreability filtering.")

    train_final = _remove_user_item_pairs(target_df, fixed_test)

    
    train_users = set(train_final["user_id"].drop_duplicates().tolist())
    train_items = set(train_final["item_id"].drop_duplicates().tolist())
    bad = fixed_test[
        ~fixed_test["user_id"].isin(train_users)
        | ~fixed_test["item_id"].isin(train_items)
    ]
    if len(bad) > 0:
        raise RuntimeError(
            f"[ADDBACK COMMON TEST] {len(bad)} fixed test pair(s) are not scoreable after final train construction."
        )

    return train_final.reset_index(drop=True), fixed_test.reset_index(drop=True), {
        "initial_test_rows": int(len(test0)),
        "fixed_test_rows": int(len(fixed_test)),
        "dropped_cold_start_test_rows": int(dropped),
    }


def _build_fixed_candidate_payload(
    target_train_df: pd.DataFrame,
    fixed_test_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_k: int,
    target_k: int,
    seed: int,
):


    import hashlib

    target_train_df = _deduplicate_latest(normalize_cols_any(target_train_df))
    fixed_test_df = _deduplicate_latest(normalize_cols_any(fixed_test_df))
    source_df = _deduplicate_latest(normalize_cols_any(source_df))

    source_train_df = _remove_user_item_pairs(source_df, fixed_test_df)

    candidate_universe = sorted(str(x) for x in target_train_df["item_id"].drop_duplicates().tolist())
    candidate_universe_set = set(candidate_universe)
    if not candidate_universe:
        raise RuntimeError("[ADDBACK COMMON TEST] Empty fixed candidate universe.")

    neg_sample = _env_int(
        "FIXED_EVAL_NEGATIVE_CANDIDATE_SAMPLE",
        _env_int("NEGATIVE_CANDIDATE_SAMPLE", 1000),
    )

    source_hist = {
        str(uid): set(str(x) for x in g["item_id"].tolist())
        for uid, g in source_train_df.groupby("user_id", sort=False)
    }

    users = {}
    skipped_no_negatives = 0
    skipped_not_scoreable = 0

    for uid, g in fixed_test_df.groupby("user_id", sort=False):
        uid_s = str(uid)
        positives = []
        for x in g["item_id"].tolist():
            x_s = str(x)
            if x_s in candidate_universe_set and x_s not in positives:
                positives.append(x_s)

        if not positives:
            skipped_not_scoreable += 1
            continue

        exclude = set(source_hist.get(uid_s, set())) | set(positives)
        neg_pool = [iid for iid in candidate_universe if iid not in exclude]

        if neg_sample and neg_sample > 0:
            n_take = min(int(neg_sample), len(neg_pool))
        else:
            n_take = len(neg_pool)

        if n_take <= 0:
            skipped_no_negatives += 1
            continue

        
        digest = hashlib.md5(uid_s.encode("utf-8")).hexdigest()
        user_seed = (int(seed) + int(digest[:8], 16)) % (2**32 - 1)
        rng = np.random.default_rng(user_seed)

        if n_take < len(neg_pool):
            negatives = rng.choice(np.asarray(neg_pool, dtype=object), size=n_take, replace=False).tolist()
        else:
            negatives = list(neg_pool)
        rng.shuffle(negatives)

        candidates = list(dict.fromkeys(negatives + positives))
        rng.shuffle(candidates)

        users[uid_s] = {
            "positives": positives,
            "candidates": [str(x) for x in candidates],
        }

    if not users:
        raise RuntimeError("[ADDBACK COMMON TEST] No users left in fixed candidate task.")

    payload = {
        "format": "fixed_candidates_v1",
        "experiment": "kcore_addback_fixed_common_test",
        "source_k": int(source_k),
        "target_k": int(target_k),
        "seed": int(seed),
        "negative_candidate_sample": int(neg_sample),
        "candidate_universe": candidate_universe,
        "users": users,
        "meta": {
            "n_eval_users": int(len(users)),
            "n_candidate_universe_items": int(len(candidate_universe)),
            "n_total_candidates": int(sum(len(v["candidates"]) for v in users.values())),
            "skipped_users_no_negatives": int(skipped_no_negatives),
            "skipped_users_not_scoreable": int(skipped_not_scoreable),
            "negative_exclusion_graph": f"kcore{int(source_k)} minus fixed target test pairs",
            "candidate_universe_definition": f"items present in kcore{int(target_k)} anchor train",
        },
    }
    return payload


def _write_shared_fixed_candidates(payload: dict, source_k: int, target_k: int) -> Path:
    out_dir = COMMON_SPLITS_DIR / "_fixed_eval" / f"addback_kcore{int(target_k)}_from_kcore{int(source_k)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fixed_candidates.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(
        f"[ADDBACK COMMON TEST] Fixed candidates saved: {out_path} | "
        f"users={payload.get('meta', {}).get('n_eval_users')} | "
        f"candidate_universe={payload.get('meta', {}).get('n_candidate_universe_items')} | "
        f"total_candidates={payload.get('meta', {}).get('n_total_candidates')}"
    )
    return out_path


def _save_split_with_fixed_test_csv(
    csv_path: str,
    variant_name: str,
    fixed_test_df: pd.DataFrame,
    force: bool = False,
    fixed_candidates_path: Path = None,
):


    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Variant CSV not found: {csv_path}")

    paths = _common_split_paths(variant_name)
    split_dir = paths["dir"]
    split_dir.mkdir(parents=True, exist_ok=True)
    fixed_ref = split_dir / "fixed_candidates.json"

    if (
        (not force)
        and paths["train_csv"].exists()
        and paths["test_csv"].exists()
        and paths["meta_json"].exists()
        and ((fixed_candidates_path is None) or fixed_ref.exists())
    ):
        print(f"[SPLIT][SKIP] Already exists: {variant_name}")
        return paths

    dfv = read_csv_smart(csv_path)
    dfv = normalize_cols_any(dfv)
    dfv = _deduplicate_latest(dfv)
    dfv = dfv[_interaction_keep_cols(dfv)].copy()

    test_df = normalize_cols_any(fixed_test_df)
    test_df = _deduplicate_latest(test_df)
    test_df = test_df[_interaction_keep_cols(test_df)].copy()

    df_keys = dfv[["user_id", "item_id"]].drop_duplicates().copy()
    df_keys["__present"] = 1

    test_check = test_df.merge(df_keys, on=["user_id", "item_id"], how="left")
    missing = int(test_check["__present"].isna().sum())
    if missing > 0:
        raise RuntimeError(
            f"[ADDBACK COMMON TEST] {variant_name}: fixed target test contains "
            f"{missing} user-item pairs that are not present in this variant. "
            "Every add-back variant should contain the target k-core interactions."
        )

    train_df = _remove_user_item_pairs(dfv, test_df)

    if fixed_candidates_path is not None:
        
        with open(fixed_candidates_path, "r", encoding="utf-8") as f:
            cand_payload = json.load(f)
        candidate_items = set(str(x) for x in cand_payload.get("candidate_universe", []))
        candidate_users = set(str(x) for x in cand_payload.get("users", {}).keys())
        train_items = set(str(x) for x in train_df["item_id"].drop_duplicates().tolist())
        train_users = set(str(x) for x in train_df["user_id"].drop_duplicates().tolist())
        missing_items = candidate_items - train_items
        missing_users = candidate_users - train_users
        if missing_items or missing_users:
            raise RuntimeError(
                f"[ADDBACK COMMON TEST] {variant_name}: fixed candidate task is not scoreable. "
                f"missing_users={len(missing_users)} missing_items={len(missing_items)}"
            )
        _write_fixed_candidates_reference(variant_name, fixed_candidates_path)

    train_df.to_csv(paths["train_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)

    meta = {
        "variant": str(variant_name),
        "source_csv": str(csv_path),
        "split_type": "fixed_target_test_with_fixed_candidates",
        "fixed_test_source": "target_kcore_plain_anchor",
        "fixed_candidates_path": str(Path(fixed_candidates_path).resolve()) if fixed_candidates_path is not None else None,
        "has_validation": False,
        "rows_total": int(len(dfv)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "users_total": int(dfv["user_id"].nunique()),
        "users_in_train": int(train_df["user_id"].nunique()),
        "users_in_test": int(test_df["user_id"].nunique()),
        "items_total": int(dfv["item_id"].nunique()),
        "items_in_train": int(train_df["item_id"].nunique()),
        "items_in_test": int(test_df["item_id"].nunique()),
        "note": (
            "Fixed evaluation split. Train is the variant minus fixed target test pairs. "
            "Ranking evaluation must use the fixed candidate file, not per-variant negative sampling."
        ),
    }

    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"[SPLIT][FIXED-TEST] {variant_name}: "
        f"train_rows={len(train_df):,} | test_rows={len(test_df):,} | "
        f"train_users={train_df['user_id'].nunique():,} | test_users={test_df['user_id'].nunique():,}"
    )
    print(f"-> Saved fixed-test split: {split_dir}")

    return paths


def _build_addback_splits_with_target_common_test(
    source_k: int,
    target_k: int,
    force: bool = False,
):


    source_k = int(source_k)
    target_k = int(target_k)

    target_name = f"kcore{target_k}_plain"
    target_csv = Path(VARIANTS_DIR) / f"ratings_{target_name}.csv"

    if not target_csv.exists():
        raise FileNotFoundError(f"Target KC{target_k} CSV not found: {target_csv}")

    split_seed = _env_int("COMMON_SPLIT_SEED", SEED)
    split_shuffle = _env_bool("COMMON_SPLIT_SHUFFLE", False)

    target_df = _deduplicate_latest(normalize_cols_any(read_csv_smart(target_csv)))
    source_df = _load_or_build_kcore_plain_for_addback(source_k)

    source_pairs = set(zip(source_df["user_id"], source_df["item_id"]))
    target_pairs = set(zip(target_df["user_id"], target_df["item_id"]))
    if not target_pairs.issubset(source_pairs):
        raise RuntimeError(
            f"[ADDBACK COMMON TEST] Expected kcore{target_k} to be nested inside kcore{source_k}, "
            "but target has interactions outside source."
        )

    target_train_df, fixed_test_df, target_split_meta = _make_fixed_target_test_split(
        target_df=target_df,
        seed=split_seed,
        shuffle=split_shuffle,
    )

    fixed_payload = _build_fixed_candidate_payload(
        target_train_df=target_train_df,
        fixed_test_df=fixed_test_df,
        source_df=source_df,
        source_k=source_k,
        target_k=target_k,
        seed=split_seed,
    )
    fixed_candidates_path = _write_shared_fixed_candidates(
        fixed_payload,
        source_k=source_k,
        target_k=target_k,
    )

    
    target_paths = _save_split_with_fixed_test_csv(
        csv_path=str(target_csv),
        variant_name=target_name,
        fixed_test_df=fixed_test_df,
        force=force,
        fixed_candidates_path=fixed_candidates_path,
    )

    
    try:
        with open(target_paths["meta_json"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["anchor_target_split_meta"] = target_split_meta
        with open(target_paths["meta_json"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Could not update target split meta with anchor details: {repr(e)}")

    pattern = f"ratings_kcore{target_k}_addback_from_kcore{source_k}_p*_seed*.csv"
    csv_paths = sorted(Path(VARIANTS_DIR).glob(pattern))

    if not csv_paths:
        print(f"[ADDBACK COMMON TEST] No add-back variant files found for pattern: {pattern}")
        return [target_paths]

    print(
        f"[ADDBACK COMMON TEST] Using fixed {target_name} task for "
        f"{len(csv_paths)} add-back variant file(s)."
    )

    out = [target_paths]
    for csv_path in csv_paths:
        variant_name = _variant_name_from_csv_path(csv_path)
        out.append(
            _save_split_with_fixed_test_csv(
                csv_path=str(csv_path),
                variant_name=variant_name,
                fixed_test_df=fixed_test_df,
                force=force,
                fixed_candidates_path=fixed_candidates_path,
            )
        )

    return out


if RUN_PREPARE_VARIANTS:
    prep_item = _env_bool("PREP_ITEM_VARIANTS", True)
    prep_user = _env_bool("PREP_USER_VARIANTS", True)
    save_raw_full = _env_bool("SAVE_RAW_FULL", False)

    prep_only_addback = _env_bool("PREP_ONLY_ADDBACK_VARIANTS", False)
    prep_addback = _env_bool("PREP_ADDBACK_VARIANTS", False)

    prep_kcore = _env_bool("PREP_KCORE_VARIANTS", False)
    kcore_fixed_user_size = _env_bool("KCORE_FIXED_USER_SIZE", False)
    kcore_align_items = False  
    kcore_save_raw_baseline = _env_bool("KCORE_SAVE_RAW_BASELINE", False)
    kcore_save_plain = _env_bool("KCORE_SAVE_PLAIN", True)
    kcore_save_aligned = _env_bool("KCORE_SAVE_ALIGNED", False)

    addback_source_k = _env_int("ADDBACK_SOURCE_K", 5)
    addback_target_k = _env_int("ADDBACK_TARGET_K", 50)
    addback_pcts = _parse_float_list(os.getenv("ADDBACK_PCTS", "0.1,0.25,0.5,1.0"), [0.1, 0.25, 0.5, 1.0])
    addback_seeds = _parse_int_list(os.getenv("ADDBACK_SEEDS", str(SEED)), [SEED])
    addback_save_target_plain = _env_bool("ADDBACK_SAVE_TARGET_PLAIN", True)
    addback_save_source_plain = _env_bool("ADDBACK_SAVE_SOURCE_PLAIN", False)

    prep_only_taildrop_stress = _env_bool("PREP_ONLY_TAILDROP_STRESS", False)
    prep_taildrop_stress = _env_bool("PREP_TAILDROP_STRESS_VARIANTS", False)

    prep_only_kcore_size_match = _env_bool("PREP_ONLY_KCORE_SIZE_MATCH", False)
    prep_kcore_size_match = _env_bool("PREP_KCORE_SIZE_MATCHED_VARIANTS", False)
    kcore_size_match_target = _env_int("KCORE_SIZE_MATCH_TARGET", 50)
    kcore_size_match_base_ks = _parse_int_list(os.getenv("KCORE_SIZE_MATCH_BASE_KS", "5,10"), [5, 10])
    kcore_size_match_seeds = _parse_int_list(os.getenv("KCORE_SIZE_MATCH_SEEDS", "42,43,44"), [42, 43, 44])
    kcore_size_match_include_target = _env_bool("KCORE_SIZE_MATCH_INCLUDE_TARGET", True)
    kcore_size_match_max_attempts = _env_int("KCORE_SIZE_MATCH_MAX_ATTEMPTS", 200)

    prep_randdrop = _env_bool("PREP_RANDOM_DROP_VARIANTS", False)

    only_modes = [
        name for name, enabled in [
            ("PREP_ONLY_ADDBACK_VARIANTS", prep_only_addback),
            ("PREP_ONLY_KCORE_SIZE_MATCH", prep_only_kcore_size_match),
            ("PREP_ONLY_TAILDROP_STRESS", prep_only_taildrop_stress),
        ]
        if enabled
    ]
    if len(only_modes) > 1:
        raise RuntimeError(f"Only one PREP_ONLY_* mode can be enabled at a time: {only_modes}")

    if prep_only_addback:
        prep_item = False
        prep_user = False
        save_raw_full = False
        prep_kcore = False
        prep_randdrop = False
        prep_taildrop_stress = False
        prep_kcore_size_match = False
        prep_addback = True

    if prep_only_kcore_size_match:
        prep_item = False
        prep_user = False
        save_raw_full = False
        prep_kcore = False
        prep_randdrop = False
        prep_taildrop_stress = False
        prep_addback = False
        prep_kcore_size_match = True

    if prep_only_taildrop_stress:
        prep_item = False
        prep_user = False
        save_raw_full = False
        prep_kcore = False
        prep_randdrop = False
        prep_addback = False
        prep_kcore_size_match = False
        prep_taildrop_stress = True
    randdrop_base_spec = os.getenv("RANDDROP_BASE", "raw").strip()
    randdrop_pcts = _parse_float_list(os.getenv("RANDDROP_PCTS", "0.2,0.4,0.6"), [0.2, 0.4, 0.6])
    randdrop_min_user = _env_int("RANDDROP_MIN_USER", 3)
    randdrop_min_item = _env_int("RANDDROP_MIN_ITEM", 2)

    target_users = _env_int("TARGET_USERS", 20000)
    target_items = _env_int("TARGET_ITEMS", 5000)

    item_head_drop = _env_float("ITEM_HEAD_DROP_PCT", 0.10)
    item_tail_drop = _env_float("ITEM_TAIL_DROP_PCT", 0.40)
    taildrop_stress_pcts = _parse_float_list(os.getenv("TAILDROP_STRESS_PCTS", "0.4,0.6,0.8"), [0.4, 0.6, 0.8])

    user_head_drop = _env_float("USER_HEAD_DROP_PCT", 0.10)
    user_tail_drop = _env_float("USER_TAIL_DROP_PCT", 0.40)

    kcore_ks_raw = os.getenv("KCORE_KS", "5,10,50").strip()
    kcore_ks = []
    if prep_kcore:
        for part in kcore_ks_raw.replace(";", ",").split(","):
            p = part.strip()
            if not p:
                continue
            try:
                kcore_ks.append(int(p))
            except Exception:
                pass
        kcore_ks = sorted(set([k for k in kcore_ks if k > 0]))
        if not kcore_ks:
            kcore_ks = [5, 10, 50]

    required = []

    def _kcore_csv_path(k: int) -> str:
        return str(VARIANTS_DIR / f"ratings_kcore{k}.csv")

    def _kcore_plain_csv_path(k: int) -> str:
        return str(VARIANTS_DIR / f"ratings_kcore{k}_plain.csv")

    def _addback_csv_path(pct: float, seed: int) -> str:
        nm = _addback_variant_name(addback_target_k, addback_source_k, pct, seed)
        return str(VARIANTS_DIR / f"ratings_{nm}.csv")

    def _kcore_size_match_csv_path(source_k: int, target_k: int, sm_seed: int) -> str:
        return str(VARIANTS_DIR / f"ratings_kcore{int(source_k)}_sizematch_kcore{int(target_k)}_seed{int(sm_seed)}.csv")

    def _taildrop_stress_csv_path(pct: float) -> str:
        return str(VARIANTS_DIR / f"ratings_{_taildrop_variant_name(float(pct))}.csv")

    kcore_required = []
    if prep_kcore:
        if kcore_save_aligned:
            kcore_required += [_kcore_csv_path(k) for k in kcore_ks]
        if kcore_save_plain:
            kcore_required += [_kcore_plain_csv_path(k) for k in kcore_ks]
        if kcore_save_raw_baseline:
            kcore_required += [str(VARIANTS_DIR / "ratings_raw_kcore_aligned.csv")]

    addback_required = []
    if prep_addback:
        for pct in addback_pcts:
            for seed_ab in addback_seeds:
                addback_required.append(_addback_csv_path(pct, seed_ab))
        if addback_save_target_plain:
            addback_required.append(_kcore_plain_csv_path(addback_target_k))
        if addback_save_source_plain:
            addback_required.append(_kcore_plain_csv_path(addback_source_k))

    size_match_required = []
    if prep_kcore_size_match:
        if kcore_size_match_include_target:
            size_match_required.append(_kcore_plain_csv_path(kcore_size_match_target))
        for source_k in kcore_size_match_base_ks:
            if int(source_k) == int(kcore_size_match_target):
                continue
            for sm_seed in kcore_size_match_seeds:
                size_match_required.append(_kcore_size_match_csv_path(source_k, kcore_size_match_target, sm_seed))

    taildrop_required = []
    if prep_taildrop_stress:
        for pct in taildrop_stress_pcts:
            taildrop_required.append(_taildrop_stress_csv_path(pct))

    if prep_item:
        required += [raw_aligned_csv, sparse_csv, dense_csv]
    if prep_user:
        required += [raw_aligned_fixitems_csv, user_sparse_csv, user_dense_csv]
    if save_raw_full:
        required += [raw_full_csv]
    if prep_kcore:
        required += kcore_required
    if prep_addback:
        required += addback_required
    if prep_kcore_size_match:
        required += size_match_required
    if prep_taildrop_stress:
        required += taildrop_required

    
    if prep_randdrop:
        for pct in randdrop_pcts:
            nm = f"randdrop{int(round(float(pct) * 100))}"
            required.append(str(VARIANTS_DIR / f"ratings_{nm}.csv"))

    if not required:
        print("RUN_PREPARE_VARIANTS=1, but no variant family is enabled.")
    else:
        need_build = not all(Path(p).exists() for p in required)

        if need_build:
            print("Building variants...")

            base = read_interactions_csv(INPUT_RATINGS_PATH)
            quick_stats(base, f"raw ({DATASET_NAME})")

            base_imp, used_thr = maybe_binarize(
                base,
                enabled=BINARIZE_FROM_EXPLICIT,
                ensure_min_pos_user=IMPLICIT_MIN_POS_PER_USER,
            )
            print(f"Variant build implicit prep: used_thr={used_thr} rows={len(base_imp):,}")
            quick_stats(base_imp, "base_imp")
            
            
            lastfm_item_cap = _env_int("LASTFM_GLOBAL_ITEM_CAP", 0)

            if DATASET_NAME.lower() == "lastfm" and lastfm_item_cap > 0:
                uniq_items = base_imp["item_id"].drop_duplicates().to_numpy()
                n_uniq_items = len(uniq_items)

                if n_uniq_items > lastfm_item_cap:
                    rng = np.random.default_rng(SEED)
                    keep_items = set(rng.choice(uniq_items, size=lastfm_item_cap, replace=False))

                    base_imp = (
                        base_imp[base_imp["item_id"].isin(keep_items)]
                        .copy()
                        .reset_index(drop=True)
                    )

                    print(
                        f"[LASTFM ITEM CAP] sampled_items={lastfm_item_cap:,} | "
                        f"users={base_imp['user_id'].nunique():,} | "
                        f"items={base_imp['item_id'].nunique():,} | "
                        f"rows={len(base_imp):,}"
                    )
                else:
                    print(
                        f"[LASTFM ITEM CAP] skipped, item_count={n_uniq_items:,} <= {lastfm_item_cap:,}"
                    )

                quick_stats(base_imp, "base_imp_after_lastfm_item_cap")

            raw_full = base_imp.copy()

            if save_raw_full:
                save_variant(raw_full, "raw_full")

            
            if prep_item:
                sparse_full = drop_items_by_pop_percentile(raw_full, top_pct=item_head_drop, bottom_pct=None)
                dense_full = drop_items_by_pop_percentile(raw_full, top_pct=None, bottom_pct=item_tail_drop)

                quick_stats(sparse_full, f"sparse_full (item head-drop {item_head_drop:.2f})")
                quick_stats(dense_full, f"dense_full (item tail-drop {item_tail_drop:.2f})")

                raw_aligned, sparse_aligned, dense_aligned = sample_users_aligned(
                    raw_full,
                    sparse_full,
                    dense_full,
                    target_users=target_users,
                    seed=SEED,
                )

                quick_stats(raw_aligned, "raw_aligned_after_sampling")
                quick_stats(sparse_aligned, "sparse_aligned_after_sampling")
                quick_stats(dense_aligned, "dense_aligned_after_sampling")

                save_variant(raw_aligned, "raw_aligned")
                save_variant(sparse_aligned, "sparse")
                save_variant(dense_aligned, "dense")

            if prep_taildrop_stress:
                pcts = [float(x) for x in taildrop_stress_pcts if float(x) > 0]
                if not pcts:
                    print("[TAILDROP STRESS][WARN] Empty TAILDROP_STRESS_PCTS; skipping stress variants.")
                else:
                    print(f"\n[TAILDROP STRESS] Building fixed-users tail-drop stress variants: pcts={pcts}")
                    tail_full = {}
                    common_users = set(raw_full["user_id"].drop_duplicates().tolist())
                    for pct in pcts:
                        df_tail = drop_items_by_pop_percentile(raw_full, top_pct=None, bottom_pct=pct)
                        df_tail = _deduplicate_latest(normalize_cols_any(df_tail))
                        tail_full[float(pct)] = df_tail
                        common_users &= set(df_tail["user_id"].drop_duplicates().tolist())
                        quick_stats(df_tail, f"taildrop_full_{_pct_label(pct)}")

                    if not common_users:
                        raise RuntimeError("[TAILDROP STRESS] No common users remain across requested tail-drop variants.")

                    rng = np.random.default_rng(SEED)
                    if target_users and len(common_users) > int(target_users):
                        chosen_users = set(
                            rng.choice(np.asarray(list(common_users), dtype=object), size=int(target_users), replace=False).tolist()
                        )
                    else:
                        chosen_users = set(common_users)

                    print(f"[TAILDROP STRESS] Fixed user set size: {len(chosen_users):,}")
                    for pct in pcts:
                        nm = _taildrop_variant_name(pct)
                        out = tail_full[float(pct)][tail_full[float(pct)]["user_id"].isin(chosen_users)].copy()
                        out = _deduplicate_latest(normalize_cols_any(out))
                        quick_stats(out, nm)
                        save_variant(out, nm)

            
            if prep_user:
                user_pop = raw_full["user_id"].value_counts()
                n_users = int(user_pop.shape[0])

                k_top_users = max(1, int(round(n_users * user_head_drop)))
                head_users = user_pop.sort_values(ascending=False).head(k_top_users).index
                user_sparse_full = raw_full[~raw_full["user_id"].isin(head_users)].copy()

                k_bot_users = max(1, int(round(n_users * user_tail_drop)))
                tail_users = user_pop.sort_values(ascending=True).head(k_bot_users).index
                user_dense_full = raw_full[~raw_full["user_id"].isin(tail_users)].copy()

                quick_stats(user_sparse_full, f"user_sparse_full (user head-drop {user_head_drop:.2f})")
                quick_stats(user_dense_full, f"user_dense_full (user tail-drop {user_tail_drop:.2f})")

                
                common_items_0 = (
                    set(raw_full["item_id"].unique())
                    & set(user_sparse_full["item_id"].unique())
                    & set(user_dense_full["item_id"].unique())
                )
                raw_u = raw_full[raw_full["item_id"].isin(common_items_0)].copy()
                user_sparse_u = user_sparse_full[user_sparse_full["item_id"].isin(common_items_0)].copy()
                user_dense_u = user_dense_full[user_dense_full["item_id"].isin(common_items_0)].copy()

                
                raw_u_fixitems, user_sparse_fixitems, user_dense_fixitems = sample_items_aligned(
                    raw_u,
                    user_sparse_u,
                    user_dense_u,
                    target_items=target_items,
                    seed=SEED,
                )

                quick_stats(raw_u_fixitems, "raw_fixitems_after_sampling")
                quick_stats(user_sparse_fixitems, "user_sparse_after_sampling")
                quick_stats(user_dense_fixitems, "user_dense_after_sampling")

                save_variant(raw_u_fixitems, "raw_aligned_fixitems")
                save_variant(user_sparse_fixitems, "user_sparse")
                save_variant(user_dense_fixitems, "user_dense")

            
            if prep_kcore:
                if not kcore_ks:
                    print("[KCORE][WARN] PREP_KCORE_VARIANTS=1, but KCORE_KS is empty. Using default values: 5,10,50")
                    kcore_ks = [5, 10, 50]

                print(f"\n[KCORE] Building k-core variants: ks={kcore_ks} | save_plain={kcore_save_plain} | save_aligned={kcore_save_aligned} | align_items={kcore_align_items} | save_raw_baseline={kcore_save_raw_baseline}")

                base0 = _deduplicate_latest(normalize_cols_any(raw_full))

                
                kcore_full = {}
                for k in kcore_ks:
                    dfk = filter_by_activity_iterative(base0, min_user=int(k), min_item=int(k))
                    quick_stats(dfk, f"kcore_full_k{k}")
                    kcore_full[int(k)] = dfk

                
                if kcore_save_plain:
                    for k in kcore_ks:
                        dfk_plain = kcore_full[int(k)].copy()
                        quick_stats(dfk_plain, f"kcore{k}_plain")
                        save_variant(dfk_plain, f"kcore{k}_plain")

                
                if kcore_save_aligned or kcore_save_raw_baseline:
                    user_sets = [set(df["user_id"].unique()) for df in kcore_full.values()]
                    common_users = set.intersection(*user_sets) if user_sets else set()

                    if not common_users:
                        print("[KCORE][WARN] No common users across k-core variants. Skipping aligned k-core export.")
                    else:
                        rng = np.random.default_rng(SEED)
                        if target_users and len(common_users) > int(target_users):
                            chosen_users = set(rng.choice(list(common_users), size=int(target_users), replace=False))
                        else:
                            chosen_users = set(common_users)

                        common_items = None

                        
                        for it in range(1, 6):
                            aligned_tmp = {
                                kk: df[df["user_id"].isin(chosen_users)].copy()
                                for kk, df in kcore_full.items()
                            }

                            if kcore_align_items:
                                item_sets = [set(df["item_id"].unique()) for df in aligned_tmp.values()]
                                common_items_new = set.intersection(*item_sets) if item_sets else set()
                                aligned_tmp = {
                                    kk: df[df["item_id"].isin(common_items_new)].copy()
                                    for kk, df in aligned_tmp.items()
                                }
                            else:
                                common_items_new = None

                            user_sets2 = [set(df["user_id"].unique()) for df in aligned_tmp.values()]
                            chosen_users_new = set.intersection(*user_sets2) if user_sets2 else set()

                            if kcore_align_items:
                                print(f"[KCORE ALIGN] iter={it} users={len(chosen_users_new):,} items={len(common_items_new):,}")
                            else:
                                print(f"[KCORE ALIGN] iter={it} users={len(chosen_users_new):,}")

                            stable_users = (chosen_users_new == chosen_users)
                            stable_items = (not kcore_align_items) or (common_items_new == common_items)

                            chosen_users = set(chosen_users_new)
                            common_items = common_items_new

                            if not chosen_users:
                                print("[KCORE][WARN] Alignment produced an empty user set. Skipping k-core export.")
                                break

                            if stable_users and stable_items:
                                break

                        if chosen_users:
                            
                            if kcore_save_aligned:
                                for k in kcore_ks:
                                    dfk = kcore_full[int(k)]
                                    dfk = dfk[dfk["user_id"].isin(chosen_users)].copy()
                                    if kcore_align_items and common_items is not None:
                                        dfk = dfk[dfk["item_id"].isin(common_items)].copy()
                                    quick_stats(dfk, f"kcore{k}_aligned")
                                    save_variant(dfk, f"kcore{k}")

                            if kcore_save_raw_baseline:
                                raw_k = base0[base0["user_id"].isin(chosen_users)].copy()
                                if kcore_align_items and common_items is not None:
                                    raw_k = raw_k[raw_k["item_id"].isin(common_items)].copy()
                                quick_stats(raw_k, "raw_kcore_aligned")
                                save_variant(raw_k, "raw_kcore_aligned")

            if prep_kcore_size_match:
                target_k = int(kcore_size_match_target)
                base_ks = sorted(set(int(k) for k in kcore_size_match_base_ks if int(k) > 0 and int(k) != target_k))
                sm_seeds = [int(x) for x in kcore_size_match_seeds]

                if not base_ks:
                    raise RuntimeError("[KCORE SIZE MATCH] Empty source k list. Set KCORE_SIZE_MATCH_BASE_KS=5,10")
                if not sm_seeds:
                    raise RuntimeError("[KCORE SIZE MATCH] Empty sampling seed list. Set KCORE_SIZE_MATCH_SEEDS=42,43,44")

                print(
                    f"\n[KCORE SIZE MATCH] target=kcore{target_k} | "
                    f"base_ks={base_ks} | seeds={sm_seeds} | "
                    "match=user+item counts only; user identity is not fixed"
                )

                base0 = _deduplicate_latest(normalize_cols_any(raw_full))
                target_df = filter_by_activity_iterative(base0, min_user=target_k, min_item=target_k)
                if target_df.empty:
                    raise RuntimeError(f"[KCORE SIZE MATCH] Target kcore{target_k} is empty.")

                target_user_count = int(target_df["user_id"].nunique())
                target_item_count = int(target_df["item_id"].nunique())
                target_rows = int(len(target_df))
                target_user_set = set(target_df["user_id"].drop_duplicates().tolist())
                target_item_set = set(target_df["item_id"].drop_duplicates().tolist())
                quick_stats(target_df, f"kcore{target_k}_target_plain")
                print(
                    f"[KCORE SIZE MATCH] target sizes: users={target_user_count:,} | "
                    f"items={target_item_count:,} | rows={target_rows:,}"
                )

                if kcore_size_match_include_target:
                    save_variant(target_df, f"kcore{target_k}_plain")
                    _write_size_match_meta(
                        f"kcore{target_k}_plain",
                        {
                            "variant": f"kcore{target_k}_plain",
                            "role": "target",
                            "target_k": target_k,
                            "users": target_user_count,
                            "items": target_item_count,
                            "rows": target_rows,
                            "note": "Target k-core saved for direct comparison. Size-matched controls match only this user/item count, not the same user IDs.",
                        },
                    )

                for source_k in base_ks:
                    source_df = filter_by_activity_iterative(base0, min_user=int(source_k), min_item=int(source_k))
                    if source_df.empty:
                        print(f"[KCORE SIZE MATCH][WARN] kcore{source_k} is empty; skipping.")
                        continue
                    quick_stats(source_df, f"kcore{source_k}_source_before_sizematch")

                    for sm_seed in sm_seeds:
                        out_name = f"kcore{source_k}_sizematch_kcore{target_k}_seed{sm_seed}"
                        out_df, meta = _sample_to_user_item_cardinality(
                            source_df,
                            target_users=target_user_count,
                            target_items=target_item_count,
                            seed=int(sm_seed),
                            max_attempts=int(kcore_size_match_max_attempts),
                            forbidden_user_set=target_user_set,
                            forbidden_item_set=target_item_set,
                        )
                        meta.update(
                            {
                                "variant": out_name,
                                "role": "source_size_matched_control",
                                "source_k": int(source_k),
                                "target_k": int(target_k),
                                "target_rows_not_matched": int(target_rows),
                                "matched_rows": int(len(out_df)),
                                "note": "Only user and item cardinalities are matched. The exact target user set is explicitly rejected; interaction count, density, degree distribution and popularity profile are not forced to match.",
                            }
                        )
                        quick_stats(out_df, out_name)
                        save_variant(out_df, out_name)
                        _write_size_match_meta(out_name, meta)

            
            if prep_addback:
                if addback_target_k <= addback_source_k:
                    raise RuntimeError(
                        f"[ADDBACK] ADDBACK_TARGET_K must be greater than ADDBACK_SOURCE_K for a nested add-back experiment. Got source={addback_source_k}, target={addback_target_k}."
                    )

                print(
                    f"\n[ADDBACK] Building add-back variants: source=kcore{addback_source_k} target=kcore{addback_target_k} "
                    f"| pcts={addback_pcts} | seeds={addback_seeds}"
                )

                base0 = _deduplicate_latest(normalize_cols_any(raw_full))
                df_source = filter_by_activity_iterative(base0, min_user=int(addback_source_k), min_item=int(addback_source_k))
                df_target = filter_by_activity_iterative(base0, min_user=int(addback_target_k), min_item=int(addback_target_k))

                quick_stats(df_source, f"addback_source_kcore{addback_source_k}")
                quick_stats(df_target, f"addback_target_kcore{addback_target_k}")

                if addback_save_source_plain:
                    save_variant(df_source, f"kcore{addback_source_k}_plain")
                if addback_save_target_plain:
                    save_variant(df_target, f"kcore{addback_target_k}_plain")

                source_pairs = set(zip(df_source["user_id"], df_source["item_id"]))
                target_pairs = set(zip(df_target["user_id"], df_target["item_id"]))
                if not target_pairs.issubset(source_pairs):
                    raise RuntimeError(
                        f"[ADDBACK] Expected kcore{addback_target_k} to be nested inside kcore{addback_source_k}, but found target interactions outside source."
                    )

                target_keys = df_target[["user_id", "item_id"]].drop_duplicates().copy()
                target_keys["__in_target"] = 1
                df_extra = df_source.merge(target_keys, on=["user_id", "item_id"], how="left")
                df_extra = df_extra[df_extra["__in_target"].isna()].drop(columns=["__in_target"]).reset_index(drop=True)
                quick_stats(df_extra, f"addback_extra_kcore{addback_source_k}_minus_kcore{addback_target_k}")

                n_extra = len(df_extra)
                print(f"[ADDBACK] extra interactions available: {n_extra:,}")

                for pct in addback_pcts:
                    pct = float(pct)
                    pct = min(max(pct, 0.0), 1.0)
                    for seed_ab in addback_seeds:
                        nm = _addback_variant_name(addback_target_k, addback_source_k, pct, seed_ab)

                        if pct >= 1.0 - 1e-12:
                            df_ab = df_source.copy()
                        else:
                            n_take = int(round(pct * n_extra))
                            n_take = min(max(n_take, 0), n_extra)
                            if n_take <= 0:
                                df_sample = df_extra.iloc[:0].copy()
                            else:
                                rng = np.random.default_rng(int(seed_ab))
                                chosen = rng.choice(df_extra.index.to_numpy(), size=n_take, replace=False)
                                df_sample = df_extra.loc[chosen].copy()

                            df_ab = pd.concat([df_target, df_sample], ignore_index=True)
                            df_ab = _deduplicate_latest(normalize_cols_any(df_ab))

                        quick_stats(df_ab, nm)
                        save_variant(df_ab, nm)

            
            if prep_randdrop:
                mode, k = _parse_randdrop_base(randdrop_base_spec)

                base0 = _deduplicate_latest(normalize_cols_any(raw_full))

                if mode == "kcore":
                    print(f"\n[RANDDROP] Base: k-core k={k}")
                    base_dense = filter_by_activity_iterative(base0, min_user=int(k), min_item=int(k))
                else:
                    print("\n[RANDDROP] Base: raw")
                    base_dense = base0

                quick_stats(base_dense, "randdrop_base_dense_before_fix")

                base_fixed = _pick_fixed_sets(
                    base_dense,
                    target_users=target_users,
                    target_items=target_items,
                    seed=SEED,
                    min_user=randdrop_min_user,
                    min_item=randdrop_min_item,
                )
                quick_stats(base_fixed, "randdrop_base_fixed")

                if base_fixed.empty or base_fixed["user_id"].nunique() < 2 or base_fixed["item_id"].nunique() < 2:
                    raise RuntimeError("[RANDDROP] base_fixed is empty or too small. Relax the degree thresholds or target sizes.")

                fix_users = set(base_fixed["user_id"].unique())
                fix_items = set(base_fixed["item_id"].unique())
                print(f"[RANDDROP] Fixed sets: users={len(fix_users):,} items={len(fix_items):,} rows={len(base_fixed):,}")
                print(f"[RANDDROP] Constraints: min_user={randdrop_min_user} min_item={randdrop_min_item}")

                for pct in randdrop_pcts:
                    pct = float(pct)
                    nm = f"randdrop{int(round(pct * 100))}"

                    rng = np.random.default_rng(SEED + int(round(pct * 1000)) + 123)
                    n = len(base_fixed)
                    n_drop = int(round(pct * n))
                    n_drop = min(max(n_drop, 0), n)
                    kept, dropped = _safe_random_drop(
                        base_fixed=base_fixed,
                        rng=rng,
                        min_user=randdrop_min_user,
                        min_item=randdrop_min_item,
                        n_drop=n_drop,
                    )

                    _assert_fixed_sets(base_fixed, kept, tag=nm)

                    
                    eff_drop = 1.0 - (len(kept) / len(base_fixed) if len(base_fixed) else 0.0)
                    print(f"[RANDDROP] {nm}: target_drop={pct:.2f} effective_drop={eff_drop:.4f} rows={len(kept):,}")

                    save_variant(kept, nm)

            print("Variants built.")
        else:
            print("Variant files already exist - skipping build.")




prep_common_splits = _env_bool("PREP_COMMON_SPLITS", True)
if prep_common_splits:
    force_split_overwrite = _env_bool("FORCE_COMMON_SPLIT_OVERWRITE", False)
    addback_common_test = _env_bool("ADDBACK_COMMON_TEST_FROM_TARGET", False)

    if addback_common_test:
        addback_source_k_split = _env_int("ADDBACK_SOURCE_K", 5)
        addback_target_k_split = _env_int("ADDBACK_TARGET_K", 50)
        _build_addback_splits_with_target_common_test(
            source_k=addback_source_k_split,
            target_k=addback_target_k_split,
            force=force_split_overwrite,
        )
    else:
        split_glob_raw = os.getenv("COMMON_SPLIT_FILE_GLOB", "").strip()
        if split_glob_raw:
            csv_paths = []
            seen = set()
            for part in split_glob_raw.replace(";", ",").split(","):
                patt = part.strip()
                if not patt:
                    continue
                for p in sorted(Path(VARIANTS_DIR).glob(patt)):
                    ps = str(p)
                    if ps not in seen:
                        csv_paths.append(p)
                        seen.add(ps)
            _build_common_splits_for_selected_variant_csvs(csv_paths, force=force_split_overwrite)
        else:
            _build_common_splits_for_all_variant_csvs(force=force_split_overwrite)


if RUN_SPARSITY_REPORTS:
    print("\nSparsity reports")

    report_list = []
    report_glob = os.getenv("SPARSITY_REPORT_FILE_GLOB", "").strip()

    if report_glob:
        seen = set()
        for part in report_glob.replace(";", ",").split(","):
            patt = part.strip()
            if not patt:
                continue
            for p in sorted(Path(VARIANTS_DIR).glob(patt)):
                ps = str(p)
                if ps not in seen:
                    report_list.append((str(p), p.stem.replace("ratings_", "")))
                    seen.add(ps)
    else:
        
        if Path(raw_aligned_csv).exists():
            report_list.append((raw_aligned_csv, "raw_aligned"))
        if Path(sparse_csv).exists():
            report_list.append((sparse_csv, "sparse"))
        if Path(dense_csv).exists():
            report_list.append((dense_csv, "dense"))

        
        if Path(raw_aligned_fixitems_csv).exists():
            report_list.append((raw_aligned_fixitems_csv, "raw_aligned_fixitems"))
        if Path(user_sparse_csv).exists():
            report_list.append((user_sparse_csv, "user_sparse"))
        if Path(user_dense_csv).exists():
            report_list.append((user_dense_csv, "user_dense"))

        
        for p in sorted(Path(VARIANTS_DIR).glob("ratings_kcore*.csv")):
            report_list.append((str(p), p.stem.replace("ratings_", "")))

        raw_kcore = Path(VARIANTS_DIR) / "ratings_raw_kcore_aligned.csv"
        if raw_kcore.exists():
            report_list.append((str(raw_kcore), "raw_kcore_aligned"))

        
        for p in sorted(Path(VARIANTS_DIR).glob("ratings_randdrop*.csv")):
            report_list.append((str(p), p.stem.replace("ratings_", "")))

        for p in sorted(Path(VARIANTS_DIR).glob("ratings_td*.csv")):
            report_list.append((str(p), p.stem.replace("ratings_", "")))

        if Path(raw_full_csv).exists():
            report_list.append((raw_full_csv, "raw_full"))

    if not report_list:
        print("No variant files found for sparsity reports.")
    else:
        seen_reports = set()
        for pth, nm in report_list:
            if pth in seen_reports:
                continue
            seen_reports.add(pth)
            dfv = read_csv_smart(pth)
            dfv = normalize_cols_any(dfv)
            sparsity_report(dfv, nm)
