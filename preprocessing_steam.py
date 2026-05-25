import argparse
import ast
import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional

USER_KEYS = ["username", "user_name", "user", "user_id", "reviewerID", "uid"]
ITEM_KEYS = ["product_id", "item_id", "appid", "app_id", "asin", "iid"]
TIME_KEYS_NUM = ["unixReviewTime", "unix_time", "timestamp", "time", "created_at", "review_time"]
TIME_KEYS_STR = ["date", "posted", "post_date", "review_date"]

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
]


def pick_first(d: Dict, keys) -> Optional[str]:
    for key in keys:
        if key in d and d[key] is not None:
            value = str(d[key]).strip()
            if value:
                return value
    return None


def parse_timestamp(d: Dict) -> Optional[int]:
    for key in TIME_KEYS_NUM:
        if key in d and d[key] is not None:
            try:
                return int(float(d[key]))
            except Exception:
                pass

    for key in TIME_KEYS_STR:
        if key in d and d[key] is not None:
            raw = str(d[key]).strip()
            if not raw:
                continue
            value = raw.replace("Posted ", "").strip()
            if value.endswith("."):
                value = value[:-1]
            for fmt in DATE_FORMATS:
                try:
                    return int(datetime.strptime(value, fmt).timestamp())
                except Exception:
                    continue
    return None


def iter_py2_dict_records(path: Path) -> Iterator[str]:
    buf = []
    depth = 0
    in_str = False
    str_ch = ""
    escape = False
    started = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for ch in line:
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == str_ch:
                        in_str = False
                else:
                    if ch in ("'", '"'):
                        in_str = True
                        str_ch = ch
                    elif ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1

                if started:
                    buf.append(ch)

                if started and depth == 0 and not in_str:
                    rec = "".join(buf).strip()
                    buf = []
                    started = False
                    if rec:
                        yield rec


def inspect_first_records(in_path: Path, n: int = 3) -> None:
    shown = 0
    for rec in iter_py2_dict_records(in_path):
        try:
            data = ast.literal_eval(rec)
        except Exception:
            continue
        print("Keys:", sorted(list(data.keys())))
        shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("No records could be parsed. The input format may differ from the expected Python-dict log format.")


def convert_to_rating_csv(in_path: Path, out_csv: Path, db_path: Path, batch_size: int = 5000) -> None:
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute(
        """
        CREATE TABLE interactions (
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            rating REAL NOT NULL,
            PRIMARY KEY (user_id, item_id)
        );
        """
    )

    upsert_sql = """
    INSERT INTO interactions(user_id, item_id, timestamp, rating)
    VALUES (?, ?, ?, 1.0)
    ON CONFLICT(user_id, item_id) DO UPDATE SET
        timestamp = CASE
            WHEN excluded.timestamp > interactions.timestamp THEN excluded.timestamp
            ELSE interactions.timestamp
        END,
        rating = 1.0;
    """

    total = 0
    kept = 0
    skipped = 0
    batch = []

    for rec in iter_py2_dict_records(in_path):
        total += 1
        try:
            data = ast.literal_eval(rec)
        except Exception:
            skipped += 1
            continue

        user = pick_first(data, USER_KEYS)
        item = pick_first(data, ITEM_KEYS)
        if user is None or item is None:
            skipped += 1
            continue

        timestamp = parse_timestamp(data)
        if timestamp is None:
            skipped += 1
            continue

        batch.append((str(user), str(item), int(timestamp)))
        kept += 1

        if len(batch) >= batch_size:
            cur.executemany(upsert_sql, batch)
            con.commit()
            batch.clear()

        if total % 200000 == 0:
            print(f"Processed records: {total:,} | kept: {kept:,} | skipped: {skipped:,}")

    if batch:
        cur.executemany(upsert_sql, batch)
        con.commit()
        batch.clear()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo)
        writer.writerow(["user_id", "item_id", "rating", "timestamp"])
        for row in cur.execute("SELECT user_id, item_id, 1.0 AS rating, timestamp FROM interactions"):
            writer.writerow(row)

    n_rows = cur.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    n_users = cur.execute("SELECT COUNT(DISTINCT user_id) FROM interactions").fetchone()[0]
    n_items = cur.execute("SELECT COUNT(DISTINCT item_id) FROM interactions").fetchone()[0]
    con.close()

    print(f"Saved {out_csv}")
    print(f"Rows: {n_rows:,} | users: {n_users:,} | items: {n_items:,}")
    print(f"SQLite cache: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Steam review logs to an implicit rating CSV.")
    parser.add_argument("--in", dest="inp", required=True, help="Input Steam review log file.")
    parser.add_argument("--out", dest="out", required=True, help="Output rating CSV.")
    parser.add_argument("--db", dest="db", default=None, help="SQLite path. Defaults to the output path with .sqlite suffix.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=5000, help="SQLite upsert batch size.")
    parser.add_argument("--inspect", action="store_true", help="Print the keys from the first parsed records and exit.")
    args = parser.parse_args()

    in_path = Path(args.inp)
    out_csv = Path(args.out)
    db_path = Path(args.db) if args.db else out_csv.with_suffix(".sqlite")

    if args.inspect:
        inspect_first_records(in_path, n=3)
        return

    convert_to_rating_csv(in_path, out_csv, db_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
