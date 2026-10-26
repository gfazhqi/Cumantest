#!/usr/bin/env python3
"""
Unduh candle 5m historis untuk backtest Alche (jalankan di VPS yang punya internet + ccxt).

Hanya data 5m yang diunduh. Timeframe 15m / 1h / 1d diturunkan oleh backtest.py dari 5m,
jadi semua timeframe selalu konsisten.

Contoh:
    pip install ccxt pandas
    python download_data.py --exchange bybit --top 100 --days 180 --out data

Hasil: data/BTC_USDT.csv.gz, data/ETH_USDT.csv.gz, ... (kolom: time,open,high,low,close,vol)
BTC_USDT dan ETH_USDT SELALU diunduh karena dipakai sebagai leader SMT/SSMT.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

TF_MS = 300_000


def is_crypto_usdt_swap(m):
    if not isinstance(m, dict):
        return False
    if not m.get("active", True) or m.get("swap") is not True or m.get("contract") is not True:
        return False
    if str(m.get("quote", "")).upper() != "USDT" or str(m.get("settle", "")).upper() != "USDT":
        return False
    base = str(m.get("base", "")).upper()
    if "TEST" in base or any(x in base for x in ("BULL", "BEAR")) or re.search(r"\d+[LS]$", base):
        return False
    return True


def file_name(sym):
    return sym.split(":")[0].replace("/", "_") + ".csv.gz"


def fetch_all(ex, sym, since_ms, until_ms, limit, retries=5):
    rows, cursor = [], since_ms
    while cursor < until_ms:
        batch = None
        for attempt in range(retries):
            try:
                batch = ex.fetch_ohlcv(sym, "5m", since=cursor, limit=limit)
                break
            except Exception as exc:  # rate limit / network
                wait = 2 + attempt * 3
                print(f"   retry {attempt + 1}/{retries} {sym}: {str(exc)[:90]} (tunggu {wait}s)")
                time.sleep(wait)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if last <= cursor:
            break
        cursor = last + TF_MS
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vol"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    # buang candle terakhir yang masih berjalan
    df = df[df["time"] + TF_MS <= until_ms]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="bybit", help="bybit | binance | bitget | bingx")
    ap.add_argument("--top", type=int, default=100, help="jumlah pair teratas berdasar volume 24h")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=1000, help="candle per request")
    ap.add_argument("--symbols", default="", help="opsional: daftar pair dipisah koma, mis. SOL/USDT:USDT,XRP/USDT:USDT")
    ap.add_argument("--resume", action="store_true", help="lewati pair yang file-nya sudah ada")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ex = getattr(ccxt, args.exchange)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    markets = ex.load_markets()
    universe = [s for s, m in markets.items() if is_crypto_usdt_swap(m)]
    if not universe:
        sys.exit("Tidak ada market USDT swap yang ditemukan.")

    must = [s for s in universe if s.split(":")[0] in ("BTC/USDT", "ETH/USDT")]
    if args.symbols:
        chosen = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        try:
            tick = ex.fetch_tickers()
            ranked = sorted(universe, key=lambda s: float((tick.get(s) or {}).get("quoteVolume") or 0), reverse=True)
        except Exception as exc:
            print(f"fetch_tickers gagal ({exc}); pakai urutan market default")
            ranked = universe
        chosen = ranked[: args.top]
    for s in must:
        if s not in chosen:
            chosen.insert(0, s)

    until = int(time.time() * 1000)
    until -= until % TF_MS
    since = until - args.days * 86_400_000
    print(f"{args.exchange}: {len(chosen)} pair, {args.days} hari -> {out}/")

    ok = 0
    for n, sym in enumerate(chosen, 1):
        path = out / file_name(sym)
        if args.resume and path.exists():
            print(f"[{n}/{len(chosen)}] {sym}: sudah ada, lewati")
            ok += 1
            continue
        df = fetch_all(ex, sym, since, until, args.limit)
        if df is None or len(df) < 1000:
            print(f"[{n}/{len(chosen)}] {sym}: data kurang, dilewati")
            continue
        df.to_csv(path, index=False, compression="gzip")
        ok += 1
        print(f"[{n}/{len(chosen)}] {sym}: {len(df):,} candle")
    print(f"Selesai: {ok}/{len(chosen)} pair tersimpan di {out}/")


if __name__ == "__main__":
    main()
