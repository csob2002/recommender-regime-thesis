import argparse
from pathlib import Path

import pandas as pd


def preprocess_online_retail(in_csv: str, out_csv: str, session_as_user: bool = False) -> pd.DataFrame:
    df = pd.read_csv(
        in_csv,
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"},
        encoding="ISO-8859-1",
    )

    df["InvoiceNo"] = df["InvoiceNo"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df = df.dropna(subset=["InvoiceDate", "StockCode", "InvoiceNo"])
    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    if session_as_user:
        df["user_id"] = df["InvoiceNo"].astype(str)
    else:
        df = df.dropna(subset=["CustomerID"])
        df["user_id"] = df["CustomerID"].astype(str)

    df["item_id"] = df["StockCode"].astype(str)

    out = df[["user_id", "item_id", "InvoiceDate"]].copy()
    out = out.rename(columns={"InvoiceDate": "timestamp"})
    out["rating"] = 1.0
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["user_id", "item_id"], keep="last").reset_index(drop=True)
    out = out[["user_id", "item_id", "rating", "timestamp"]]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Online Retail dataset to an implicit interaction CSV.")
    parser.add_argument("--in", dest="input_csv", default="OnlineRetail.csv", help="Input Online Retail CSV file.")
    parser.add_argument("--out", dest="output_csv", default="ratings_raw_full.csv", help="Output interaction CSV file.")
    parser.add_argument("--session-as-user", action="store_true", help="Use InvoiceNo as the user identifier instead of CustomerID.")
    args = parser.parse_args()

    out = preprocess_online_retail(args.input_csv, args.output_csv, session_as_user=args.session_as_user)
    print(f"Saved {args.output_csv} | rows={len(out):,} | users={out['user_id'].nunique():,} | items={out['item_id'].nunique():,}")


if __name__ == "__main__":
    main()
