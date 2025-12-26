import os
import sys
import argparse
import math
from collections import Counter
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Config
POS_THRESHOLD = 0.15
NEG_THRESHOLD = -0.15
DEFAULT_PCT_FILL = 0.0
BATCH_UPSERT = True

# Helpers
def get_db_engine():
    DB_URL = os.getenv("DB_URL")
    if not DB_URL:
        raise Exception("Set DB_URL environment variable first (postgres connection string).")
    engine = create_engine(DB_URL, pool_pre_ping=True)
    return engine

def find_col_mapping(df_cols):
    cols = list(df_cols)
    cols_low = {c.lower(): c for c in cols}

    def find(*candidates):
        for cand in candidates:
            if cand is None:
                continue
            cl = cand.lower()
            if cl in cols_low:
                return cols_low[cl]
        return None

    mapping = {}
    c = find('date', 'tanggal', 'tgl')
    if c: mapping[c] = 'date'
    c = find('terakhir', 'last', 'close')
    if c: mapping[c] = 'terakhir'
    c = find('pembukaan', 'open', 'pembukaan')
    if c: mapping[c] = 'pembukaan'
    c = find('tertinggi', 'high', 'tertinggi')
    if c: mapping[c] = 'tertinggi'
    c = find('terendah', 'low', 'terendah')
    if c: mapping[c] = 'terendah'
    c = find('volume', 'vol.', 'vol', 'vol')
    if c: mapping[c] = 'volume'
    c = find('perubahan_pct', 'perubahan%', 'perubahan', 'perubahan_persen', 'perubahan_pc')
    if c: mapping[c] = 'perubahan_pct'
    return mapping

def normalize_prices_df(df):
    df.columns = [c.strip() for c in df.columns]

    mapping = find_col_mapping(df.columns)
    if mapping:
        df = df.rename(columns=mapping)

    if 'date' not in df.columns:
        lc = {c.lower(): c for c in df.columns}
        for candidate in ('tanggal','tgl'):
            if candidate in lc:
                df = df.rename(columns={lc[candidate]: 'date'})
                break

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.date
    else:
        df['date'] = pd.NaT

    def to_float_safe(s):
        if pd.isna(s): return np.nan
        s2 = str(s).strip()
        if s2 == '':
            return np.nan
        # remove percent sign
        if s2.endswith('%'):
            s2 = s2[:-1]
        if '.' in s2 and ',' in s2:
            s2 = s2.replace('.', '').replace(',', '.')
        else:
            if ',' in s2 and '.' not in s2:
                s2 = s2.replace(',', '.')
        s2 = "".join(ch for ch in s2 if ch.isdigit() or ch in ('.','-'))
        try:
            return float(s2)
        except:
            return np.nan

    for col in ['terakhir','pembukaan','tertinggi','terendah','perubahan_pct']:
        if col in df.columns:
            df[col] = df[col].apply(to_float_safe)

    def parse_vol(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        s = s.replace(',', '').replace(' ', '')
        try:
            if s.endswith(('B','b')):
                return float(s[:-1]) * 1e9
            if s.endswith(('M','m')):
                return float(s[:-1]) * 1e6
            if s.endswith(('K','k')):
                return float(s[:-1]) * 1e3
            return float("".join(ch for ch in s if ch.isdigit() or ch=='.' or ch=='-'))
        except:
            return np.nan

    if 'volume' in df.columns:
        df['volume'] = df['volume'].apply(parse_vol)
    elif 'vol.' in df.columns:
        df = df.rename(columns={'vol.': 'volume'})
        df['volume'] = df['volume'].apply(parse_vol)

    # drop rows without date
    before = len(df)
    df = df.dropna(subset=['date'])
    after = len(df)
    if before != after:
        print(f"[prices] dropped {before-after} rows with invalid/missing date")
    return df

def compute_signed_agg(engine):
    q = text("""
        SELECT r.tanggal::date as date, p.sentiment_label, p.sentiment_score
        FROM processed_news p
        JOIN raw_news r ON p.raw_id = r.id
        WHERE p.sentiment_label IS NOT NULL
    """)
    df = pd.read_sql(q, engine)
    if df.empty:
        print("[agg] No processed_news with sentiment found.")
        return None

    # normalize label string lower-case
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.lower()
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce').fillna(0.0)

    # signed score
    def signed(row):
        lbl = row['sentiment_label']
        sc = float(row['sentiment_score']) if not pd.isna(row['sentiment_score']) else 0.0
        if 'positif' in lbl:
            return sc
        elif 'negatif' in lbl:
            return -sc
        else:
            return 0.0

    df['signed_score'] = df.apply(signed, axis=1)

    agg = df.groupby('date').agg(
        avg_sentiment = ('signed_score', 'mean'),
        prop_pos = ('sentiment_label', lambda x: (x=='positif').mean()),
        prop_neg = ('sentiment_label', lambda x: (x=='negatif').mean()),
        max_impact = ('sentiment_label', lambda x: Counter(x).most_common(1)[0][0] if len(x)>0 else None)
    ).reset_index()

    agg['avg_sentiment'] = pd.to_numeric(agg['avg_sentiment'], errors='coerce')
    agg['prop_pos'] = pd.to_numeric(agg['prop_pos'], errors='coerce')
    agg['prop_neg'] = pd.to_numeric(agg['prop_neg'], errors='coerce')

    return agg

def merge_prices_and_agg(prices_df, agg_df):
    # Ensure both datatypes are date
    prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
    agg_df['date'] = pd.to_datetime(agg_df['date']).dt.date

    merged = prices_df.merge(agg_df, on='date', how='left')

    def compute_pct(row):
        p = row.get('perubahan_pct')
        if pd.notna(p):
            return p
        last = row.get('terakhir')
        openp = row.get('pembukaan')
        try:
            if pd.isna(last) or pd.isna(openp) or float(openp) == 0:
                return np.nan
            return ((float(last) - float(openp)) / float(openp)) * 100.0
        except:
            return np.nan

    merged['perubahan_pct'] = merged.apply(compute_pct, axis=1)
    merged['perubahan_pct'] = merged['perubahan_pct'].fillna(DEFAULT_PCT_FILL)

    merged['avg_sentiment'] = merged['avg_sentiment'].where(pd.notna(merged['avg_sentiment']), np.nan)
    merged['prop_pos'] = merged['prop_pos'].where(pd.notna(merged['prop_pos']), np.nan)
    merged['prop_neg'] = merged['prop_neg'].where(pd.notna(merged['prop_neg']), np.nan)
    merged['max_impact'] = merged['max_impact'].where(pd.notna(merged['max_impact']), 'tidak ada data')

    def categorize_signed(s):
        if pd.isna(s):
            return 'tidak ada data'
        try:
            v = float(s)
        except:
            return 'tidak ada data'
        if v >= POS_THRESHOLD:
            return 'positif'
        if v <= NEG_THRESHOLD:
            return 'negatif'
        return 'netral'

    merged['avg_sentiment_result'] = merged['avg_sentiment'].apply(categorize_signed)
    return merged

def upsert_features(engine, df_merged, dry_run=False):
    if df_merged is None or df_merged.empty:
        print("[upsert] no rows to upsert.")
        return

    print(f"[upsert] rows to upsert: {len(df_merged)}")
    if dry_run:
        print("[upsert] dry-run enabled, not writing to DB. Showing head:")
        print(df_merged.head(10))
        return

    insert_sql = text("""
        INSERT INTO ihsg_sentimen_features
        (date, terakhir, pembukaan, tertinggi, terendah, volume, perubahan_pct, avg_sentiment, prop_pos, prop_neg, max_impact, avg_sentiment_result, created_at)
        VALUES (:date, :terakhir, :pembukaan, :tertinggi, :terendah, :volume, :perubahan_pct, :avg_sentiment, :prop_pos, :prop_neg, :max_impact, :avg_sentiment_result, now())
        ON CONFLICT (date) DO UPDATE SET
          terakhir = EXCLUDED.terakhir,
          pembukaan = EXCLUDED.pembukaan,
          tertinggi = EXCLUDED.tertinggi,
          terendah = EXCLUDED.terendah,
          volume = EXCLUDED.volume,
          perubahan_pct = EXCLUDED.perubahan_pct,
          avg_sentiment = EXCLUDED.avg_sentiment,
          prop_pos = EXCLUDED.prop_pos,
          prop_neg = EXCLUDED.prop_neg,
          max_impact = EXCLUDED.max_impact,
          avg_sentiment_result = EXCLUDED.avg_sentiment_result,
          created_at = now();""")

    # Upsert in transaction
    with engine.begin() as conn:
        count = 0
        for _, row in df_merged.iterrows():
            conn.execute(insert_sql, {
                "date": row["date"],
                "terakhir": None if pd.isna(row.get("terakhir")) else float(row.get("terakhir")),
                "pembukaan": None if pd.isna(row.get("pembukaan")) else float(row.get("pembukaan")),
                "tertinggi": None if pd.isna(row.get("tertinggi")) else float(row.get("tertinggi")),
                "terendah": None if pd.isna(row.get("terendah")) else float(row.get("terendah")),
                "volume": None if pd.isna(row.get("volume")) else float(row.get("volume")),
                "perubahan_pct": float(row.get("perubahan_pct")) if not pd.isna(row.get("perubahan_pct")) else DEFAULT_PCT_FILL,
                "avg_sentiment": None if pd.isna(row.get("avg_sentiment")) else float(row.get("avg_sentiment")),
                "prop_pos": None if pd.isna(row.get("prop_pos")) else float(row.get("prop_pos")),
                "prop_neg": None if pd.isna(row.get("prop_neg")) else float(row.get("prop_neg")),
                "max_impact": row.get("max_impact"),
                "avg_sentiment_result": row.get("avg_sentiment_result")
            })
            count += 1
    print(f"[upsert] done. {count} rows upserted/updated.")

# main
def main(dry_run=False):
    engine = get_db_engine()

    # get prices
    print("[main] reading prices table...")
    prices_q = text("SELECT * FROM prices")
    prices_df = pd.read_sql(prices_q, engine)
    prices_df = normalize_prices_df(prices_df)
    if prices_df.empty:
        print("[main] prices table is empty after normalization. Exiting.")
        return

    # compute signed daily agg from processed_news
    print("[main] computing signed daily sentiment aggregation...")
    agg_df = compute_signed_agg(engine)
    if agg_df is None or agg_df.empty:
        print("[main] no aggregated sentiment rows. Exiting.")
        return

    # merge & compute features
    print("[main] merging prices and sentiment aggregation...")
    merged = merge_prices_and_agg(prices_df, agg_df)

    print(f"[main] merged rows: {len(merged)}")
    upsert_features(engine, merged, dry_run=dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Do not write to DB; just print head")
    args = parser.parse_args()
    main(dry_run=args.dry_run)