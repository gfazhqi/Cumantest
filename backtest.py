#!/usr/bin/env python3
"""
Backtest walk-forward untuk engine Alche: MNSR, A+ dan AMD/Judas.

Fungsi analisa (analyze_mnsr_full, analyze_a_plus_setup, analyze_amd_judas beserta helper-nya)
DIAMBIL LANGSUNG dari Alche.py lewat AST, bukan ditulis ulang, sehingga yang diuji adalah
kode yang sama dengan bot live.

Cara pakai:
    python download_data.py --exchange bybit --top 100 --days 180 --out data
    python backtest.py --alche Alche.py --data-dir data --days 180 --workers 4 --out results

Output di folder --out: summary.md, trades.csv, equity_R.png

Aturan simulasi (semua konservatif / tanpa look-ahead):
  * Keputusan diambil pada CLOSE candle 5m. Semua timeframe (15m/1h/1d) diturunkan dari 5m;
    candle yang belum selesai diganti candle sintetis (o=h=l=c=close terakhir), sama seperti
    candle "forming" di scanner live yang selalu dibuang oleh engine (iloc[:-1]).
  * MNSR dievaluasi per window 15m, A+ per candle 5m (hanya sesi Asia 07-16 WIB, gate CRT H1),
    AMD per candle 5m (sesi London/NY WIB). AMD memakai candle 5m yang baru close sebagai
    "last", bukan candle yang sedang berjalan seperti live (batasan: tidak ada data intrabar).
  * Order LIMIT: fill jika harga menyentuh entry dalam 60 menit (MAX_PENDING_MINUTES), jika TP
    tersentuh sebelum fill -> MISSED, tidak ada fill -> TIMEOUT. Slot tetap terpakai selama PENDING.
  * Candle yang menyentuh SL dan TP sekaligus dianggap SL. Candle fill hanya dihitung TP
    jika close >= TP. SL kena gap -> keluar di harga open candle.
  * Filter harga live: BUY butuh entry < harga saat ini (open candle berikutnya), SELL sebaliknya.
  * Biaya: maker (entry) + taker (exit) + slippage SL, dapat diatur.
  * Satu trade aktif per pair per engine. Satu sinyal MNSR per window 15m.
  * Tidak dimodelkan: filter news USD, rotasi scan 40 pair, presisi/min-notional exchange, likuidasi.
"""
import argparse
import ast
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WIB = timezone(timedelta(hours=7))
TF5 = 300_000
TF15 = 900_000
TF1H = 3_600_000
DAY = 86_400_000
COLS = ["time", "open", "high", "low", "close", "vol"]
PENDING_BARS = 12  # 60 menit = MAX_PENDING_MINUTES

NEED_FUNCS = {
    "_mnsr_swings", "_mnsr_structure", "_mnsr_fvg_ob", "_mnsr_smt_ssmt",
    "_mnsr_previous_liquidity", "analyze_mnsr_full",
    "_a_plus_asia_session", "_a_plus_h1_crt", "_a_plus_narrative_erl_irl",
    "_a_plus_m5_cisd", "analyze_a_plus_setup",
    "_amd_hour_wib", "_amd_exec_session", "_amd_asia_range", "analyze_amd_judas",
}
NEED_CONSTS = {
    "SL_BUFFER_PERCENT", "A_PLUS_ENABLED", "A_PLUS_ASIA_START_HOUR", "A_PLUS_ASIA_END_HOUR",
    "A_PLUS_MIN_RR", "A_PLUS_M5_LOOKBACK", "A_PLUS_H1_LOOKBACK",
    "AMD_ENABLED", "AMD_TF", "AMD_MIN_RR", "AMD_ASIA_START_HOUR_WIB", "AMD_ASIA_END_HOUR_WIB",
    "AMD_LONDON_START_HOUR_WIB", "AMD_LONDON_END_HOUR_WIB",
    "AMD_NEWYORK_START_HOUR_WIB", "AMD_NEWYORK_END_HOUR_WIB",
}


# ----------------------------------------------------------------------------
# Memuat engine asli dari Alche.py
# ----------------------------------------------------------------------------
def load_engine(path):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    body, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NEED_FUNCS:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & NEED_CONSTS:
                body.append(node)
                found |= names
    missing = (NEED_FUNCS | NEED_CONSTS) - found
    if missing:
        raise SystemExit(f"Alche.py tidak memuat: {sorted(missing)}")
    ns = {"pd": pd, "np": np, "datetime": datetime, "timedelta": timedelta,
          "timezone": timezone, "WIB": WIB}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    ns["_mnsr_swings_orig"] = ns["_mnsr_swings"]
    ns["_mnsr_swings"] = _mnsr_swings_fast  # hasil identik (diverifikasi), ~50x lebih cepat
    return ns


def _mnsr_swings_fast(df, lookback=3):
    """Padanan numpy dari _mnsr_swings di Alche.py (semantik sama persis)."""
    d = df.iloc[:-1] if len(df) > 1 else df
    highs, lows = [], []
    n = len(d)
    if n < (lookback * 2 + 3):
        return highs, lows
    hi = d["high"].values.astype("float64")
    lo = d["low"].values.astype("float64")
    w = 2 * lookback + 1
    win = np.lib.stride_tricks.sliding_window_view
    hmax = win(hi, w).max(axis=1)  # indeks k -> pusat i = k + lookback
    lmin = win(lo, w).min(axis=1)
    for k in range(len(hmax)):
        i = k + lookback
        if hi[i] >= hmax[k]:
            highs.append((i, float(hi[i])))
        if lo[i] <= lmin[k]:
            lows.append((i, float(lo[i])))
    return highs, lows


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df[COLS]


def aggregate(df5, tf_ms):
    key = df5["time"] - df5["time"] % tf_ms
    g = df5.groupby(key)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "vol": g["vol"].sum(), "n": g["time"].count(),
    })
    out = out[out["n"] == tf_ms // TF5]  # buang bucket tidak lengkap
    out.insert(0, "time", out.index.astype("int64"))
    return out.drop(columns="n").reset_index(drop=True)


class TF:
    """Akses window candle yang SUDAH close pada waktu t, ditambah satu candle forming sintetis."""

    def __init__(self, df, tf_ms):
        self.tf = tf_ms
        self.t = df["time"].values.astype("int64")
        self.M = df[COLS].values.astype("float64")

    def window(self, t, n):
        k = int(np.searchsorted(self.t, t - self.tf, side="right"))
        if k <= 0 or self.t[k - 1] < t - 3 * self.tf:
            return None
        w = self.M[max(0, k - n):k]
        last = w[-1, 4]
        forming = np.array([[t, last, last, last, last, 0.0]])
        return pd.DataFrame(np.vstack([w, forming]), columns=COLS)


class Sym:
    def __init__(self, df5):
        self.df5 = df5
        self.t5 = df5["time"].values.astype("int64")
        self.o5 = df5["open"].values.astype("float64")
        self.h5 = df5["high"].values.astype("float64")
        self.l5 = df5["low"].values.astype("float64")
        self.c5 = df5["close"].values.astype("float64")
        self.rows5 = df5[COLS].values.tolist()
        self.m15 = TF(aggregate(df5, TF15), TF15)
        self.h1 = TF(aggregate(df5, TF1H), TF1H)

    def day_open(self, t):
        day0 = t - t % DAY
        idx = int(np.searchsorted(self.t5, day0, side="left"))
        if idx < len(self.t5) and self.t5[idx] < t:
            return float(self.o5[idx])
        k = int(np.searchsorted(self.t5, t, side="left")) - 1
        return float(self.c5[max(k, 0)])


# ----------------------------------------------------------------------------
# Worker state
# ----------------------------------------------------------------------------
G = {}


def _init_worker(alche_path, data_dir, cfg):
    G["ns"] = load_engine(alche_path)
    G["cfg"] = cfg
    G["data_dir"] = Path(data_dir)
    leaders = {}
    for name in ("BTC_USDT", "ETH_USDT"):
        p = G["data_dir"] / f"{name}.csv.gz"
        leaders[name] = Sym(load_csv(p)) if p.exists() else None
    G["leaders"] = leaders


# ----------------------------------------------------------------------------
# Simulasi order LIMIT
# ----------------------------------------------------------------------------
def sim_trade(S, i, side, en, sl, tp):
    """Return (status, fill_idx, exit_idx, exit_price). Index = indeks candle 5m."""
    N = len(S.t5)
    h, l, c, o = S.h5, S.l5, S.c5, S.o5
    end = min(N, i + 1 + PENDING_BARS)
    jf = None
    for j in range(i + 1, end):
        if side == "BUY":
            if l[j] <= en:
                jf = j
                break
            if h[j] >= tp:
                return "NOFILL_MISSED", None, j, None
        else:
            if h[j] >= en:
                jf = j
                break
            if l[j] <= tp:
                return "NOFILL_MISSED", None, j, None
    if jf is None:
        return "NOFILL_TIMEOUT", None, max(i + 1, end - 1), None

    j = jf
    if side == "BUY":
        if l[j] <= sl:
            return "SL", jf, j, min(sl, max(o[j], 0)) if o[j] <= sl else sl
        if c[j] >= tp:
            return "TP", jf, j, tp
    else:
        if h[j] >= sl:
            return "SL", jf, j, max(sl, o[j]) if o[j] >= sl else sl
        if c[j] <= tp:
            return "TP", jf, j, tp

    for k in range(jf + 1, N):
        if side == "BUY":
            if l[k] <= sl:
                return "SL", jf, k, (o[k] if o[k] <= sl else sl)
            if h[k] >= tp:
                return "TP", jf, k, tp
        else:
            if h[k] >= sl:
                return "SL", jf, k, (o[k] if o[k] >= sl else sl)
            if l[k] <= tp:
                return "TP", jf, k, tp
    return "OPEN_END", jf, N - 1, None


def make_trade(S, sym, engine, i, side, en, sl, tp, res, cfg):
    status, jf, jx, xp = res
    t_create = int(S.t5[i] + TF5)
    risk = abs(en - sl)
    tr = {
        "engine": engine, "symbol": sym, "side": side, "t_create": t_create,
        "t_fill": int(S.t5[jf]) if jf is not None else None,
        "t_exit": int(S.t5[jx] + TF5), "entry": en, "sl": sl, "tp": tp, "exit": xp,
        "status": status, "rr_plan": abs(tp - en) / risk if risk > 0 else 0.0,
        "risk_pct": risk / en * 100.0, "gross_R": 0.0, "net_R": 0.0,
        "hold_bars": (jx - jf) if jf is not None else 0, "pnl_usd": 0.0,
    }
    if status in ("TP", "SL") and risk > 0:
        d = 1.0 if side == "BUY" else -1.0
        gross = (xp - en) * d / risk
        cost = (cfg["maker"] * en + cfg["taker"] * xp) / risk
        if status == "SL":
            cost += cfg["slip"] * xp / risk
        tr["gross_R"] = gross
        tr["net_R"] = gross - cost
        qty = cfg["margin"] * cfg["lev"] / en
        fees = qty * (cfg["maker"] * en + cfg["taker"] * xp)
        if status == "SL":
            fees += qty * cfg["slip"] * xp
        tr["pnl_usd"] = qty * (xp - en) * d - fees
    return tr


# ----------------------------------------------------------------------------
# Walk-forward satu pair
# ----------------------------------------------------------------------------
def run_symbol(job):
    sym, engines_label, engines = job
    ns, cfg, leaders = G["ns"], G["cfg"], G["leaders"]
    path = G["data_dir"] / f"{sym}.csv.gz"
    df = load_csv(path)
    last_t = int(df["time"].iloc[-1])
    first_keep = last_t - (cfg["days"] + 4) * DAY
    df = df[df["time"] >= first_keep].reset_index(drop=True)
    if len(df) < 2000:
        return sym, []
    S = Sym(df)
    btc, eth = leaders["BTC_USDT"], leaders["ETH_USDT"]
    t5, c5, o5 = S.t5, S.c5, S.o5
    N = len(t5)

    start_t = last_t - cfg["days"] * DAY
    i = max(int(np.searchsorted(t5, start_t)), 800)
    analyze_mnsr = ns["analyze_mnsr_full"]
    analyze_aplus = ns["analyze_a_plus_setup"]
    analyze_amd = ns["analyze_amd_judas"]
    crt_fn = ns["_a_plus_h1_crt"]
    asia_fn = ns["_a_plus_asia_session"]
    amd_sess = ns["_amd_exec_session"]

    mn = {"win": None, "sig": None, "used": False}
    ap = {"hour": None, "ok": False, "h1": None, "bh": None, "eh": None}
    trades = []

    while i < N - 1:
        t = int(t5[i] + TF5)
        if t5[i + 1] != t:  # gap data
            i += 1
            continue
        sig = None
        engine = None

        if "MNSR" in engines and btc is not None and eth is not None:
            win = t - t % TF15
            if mn["win"] != win:
                mn = {"win": win, "sig": None, "used": False}
                m15 = S.m15.window(win, 50)
                h1 = S.h1.window(win, 60)
                ba, ea = btc.m15.window(win, 60), eth.m15.window(win, 60)
                if m15 is not None and h1 is not None and ba is not None and ea is not None:
                    try:
                        r = analyze_mnsr(m15, h1, S.day_open(win), ba, ea)
                        if r and r[0]:
                            mn["sig"] = r
                    except Exception:
                        pass
            if mn["sig"] and not mn["used"]:
                sig, engine = mn["sig"], "MNSR"

        if sig is None and "A+" in engines and btc is not None and eth is not None:
            now = datetime.fromtimestamp(t / 1000.0, WIB)
            if asia_fn(now):
                hk = t - t % TF1H
                if ap["hour"] != hk:
                    ap = {"hour": hk, "ok": False, "h1": None, "bh": None, "eh": None}
                    h1 = S.h1.window(hk, 60)
                    if h1 is not None and len(h1) >= 30:
                        try:
                            if crt_fn(h1) is not None:
                                ap.update(ok=True, h1=h1, bh=btc.h1.window(hk, 60), eh=eth.h1.window(hk, 60))
                        except Exception:
                            pass
                if ap["ok"] and ap["bh"] is not None and ap["eh"] is not None:
                    m5 = pd.DataFrame(S.rows5[max(0, i - 38):i + 1] + [[t, c5[i], c5[i], c5[i], c5[i], 0.0]], columns=COLS)
                    try:
                        r = analyze_aplus(ap["h1"], m5, ap["bh"], ap["eh"], now=now)
                        if r and r[0]:
                            sig, engine = r, "A+"
                    except Exception:
                        pass

        if sig is None and "AMD" in engines and amd_sess(int(t5[i])):
            try:
                r = analyze_amd(S.rows5[max(0, i - 119):i + 1])
                if r and r[0]:
                    sig, engine = r, "AMD"
            except Exception:
                pass

        if sig is None:
            i += 1
            continue

        side, en, sl, tp = sig[0], float(sig[1]), float(sig[2]), float(sig[3])
        valid = en > 0 and ((sl < en < tp) if side == "BUY" else (tp < en < sl))
        live_price = float(o5[i + 1])
        if valid and cfg["price_filter"]:
            if (side == "BUY" and en >= live_price) or (side == "SELL" and en <= live_price):
                valid = False
        if not valid:
            i += 1
            continue

        res = sim_trade(S, i, side, en, sl, tp)
        trades.append(make_trade(S, sym, engine, i, side, en, sl, tp, res, cfg))
        if engine == "MNSR":
            mn["used"] = True
        i = max(res[2], i) + 1

    return sym, trades


# ----------------------------------------------------------------------------
# Statistik
# ----------------------------------------------------------------------------
def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m) * 100, min(1.0, c + m) * 100


def stats(df):
    """df = trade dari satu engine (semua status)."""
    n_all = len(df)
    f = df[df["status"].isin(["TP", "SL"])].sort_values("t_exit")
    n = len(f)
    out = {"signals": n_all, "filled_closed": n,
           "nofill": int(df["status"].str.startswith("NOFILL").sum()),
           "open_end": int((df["status"] == "OPEN_END").sum())}
    if n == 0:
        return out | {"wins": 0, "losses": 0, "wr": 0.0, "wr_lo": 0.0, "wr_hi": 0.0, "avg_R": 0.0,
                      "total_R": 0.0, "gross_R": 0.0, "pf": 0.0, "max_dd_R": 0.0, "be_wr": 0.0,
                      "avg_rr_plan": 0.0, "max_lose_streak": 0, "avg_hold_min": 0.0}
    wins = int((f["status"] == "TP").sum())
    losses = n - wins
    r = f["net_R"].values
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    cum = np.cumsum(r)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    streak = best = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    lo, hi = wilson(wins, n)
    avg_w = r[r > 0].mean() if (r > 0).any() else 0.0
    avg_l = -r[r < 0].mean() if (r < 0).any() else 0.0
    be = avg_l / (avg_w + avg_l) * 100 if (avg_w + avg_l) > 0 else 0.0
    return out | {
        "wins": wins, "losses": losses, "wr": wins / n * 100, "wr_lo": lo, "wr_hi": hi,
        "avg_R": float(r.mean()), "total_R": float(r.sum()), "gross_R": float(f["gross_R"].sum()),
        "pf": float(gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0,
        "max_dd_R": dd, "be_wr": be, "avg_rr_plan": float(f["rr_plan"].mean()),
        "max_lose_streak": best, "avg_hold_min": float(f["hold_bars"].mean() * 5),
    }


def fmt_pf(x):
    return "inf" if x == float("inf") else f"{x:.2f}"


def headline_row(name, s):
    return (f"| {name} | {s['signals']} | {s['filled_closed']} | {s['nofill']} | {s['wr']:.1f}% "
            f"({s['wr_lo']:.0f}-{s['wr_hi']:.0f}) | {s['be_wr']:.1f}% | {s['avg_R']:+.3f} | "
            f"{s['total_R']:+.1f} | {fmt_pf(s['pf'])} | {s['max_dd_R']:.1f} | {s['max_lose_streak']} |")


HEADER = ("| Engine | Sinyal | Closed | No-fill | WR % (CI95) | WR impas % | Avg R | Total R | PF | Max DD R | Rugi beruntun |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")


def group_table(df, col, min_n=1, top=None, sort="total_R"):
    f = df[df["status"].isin(["TP", "SL"])]
    rows = []
    for key, g in f.groupby(col):
        if len(g) < min_n:
            continue
        w = int((g["status"] == "TP").sum())
        rows.append((key, len(g), w / len(g) * 100, g["net_R"].mean(), g["net_R"].sum()))
    rows.sort(key=lambda x: x[4], reverse=(sort != "asc"))
    if top:
        rows = rows[:top]
    lines = ["| " + str(col) + " | N | WR % | Avg R | Total R |", "|---|---|---|---|---|"]
    for key, n, wr, avg, tot in rows:
        lines.append(f"| {key} | {n} | {wr:.1f} | {avg:+.3f} | {tot:+.1f} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Portfolio (aturan REAL: max posisi, max entry/hari, daily realized loss)
# ----------------------------------------------------------------------------
def portfolio(df, cfg):
    """Terapkan limit per CEX seperti bot REAL pada seluruh trade engine ini (urut waktu)."""
    df = df.sort_values("t_create")
    acc, active = [], []
    day_count, day_loss_events = {}, {}
    for _, tr in df.iterrows():
        tc = tr["t_create"]
        day = datetime.fromtimestamp(tc / 1000, WIB).date()
        active = [e for e in active if e > tc]
        if len(active) >= cfg["max_pos"]:
            continue
        if day_count.get(day, 0) >= cfg["max_day"]:
            continue
        realized = sum(p for te, p in day_loss_events.get(day, []) if te <= tc)
        if -realized >= cfg["max_loss"]:
            continue
        acc.append(tr)
        active.append(tr["t_exit"])
        day_count[day] = day_count.get(day, 0) + 1
        if tr["status"] in ("TP", "SL"):
            dday = datetime.fromtimestamp(tr["t_exit"] / 1000, WIB).date()
            day_loss_events.setdefault(dday, []).append((tr["t_exit"], tr["pnl_usd"]))
    return pd.DataFrame(acc)


def portfolio_text(name, p, cfg):
    f = p[p["status"].isin(["TP", "SL"])].sort_values("t_exit") if len(p) else p
    if len(f) == 0:
        return f"- **{name}**: tidak ada trade yang lolos limit.\n"
    pnl = f["pnl_usd"].values
    cum = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    liq = int((f["risk_pct"] * cfg["lev"] >= 100).sum())
    days = max(1, (f["t_exit"].max() - f["t_exit"].min()) / DAY)
    return (f"- **{name}**: {len(f)} trade closed, WR {(f['status'] == 'TP').mean() * 100:.1f}%, "
            f"net **${pnl.sum():+.2f}** (rata-rata ${pnl.sum() / days:+.3f}/hari), max drawdown ${dd:.2f}, "
            f"trade dengan SL>=1/lev dari entry (risiko likuidasi pada margin ${cfg['margin']}): {liq}\n")


# ----------------------------------------------------------------------------
# Laporan
# ----------------------------------------------------------------------------
def build_report(all_df, cfg, symbols, period):
    L = []
    L.append("# Backtest Alche — ringkasan\n")
    L.append(f"- Periode: {period}  |  Pair: {len(symbols)}  |  Data 5m diturunkan ke 15m/1h/1d\n"
             f"- Biaya: maker {cfg['maker'] * 100:.3f}% (entry), taker {cfg['taker'] * 100:.3f}% (exit), "
             f"slippage SL {cfg['slip'] * 100:.3f}%\n"
             f"- Filter harga live: {'ON' if cfg['price_filter'] else 'OFF'}  |  "
             f"Portfolio: margin ${cfg['margin']}, {cfg['lev']}x, max {cfg['max_pos']} posisi, "
             f"{cfg['max_day']} entry/hari, daily loss ${cfg['max_loss']}\n")
    L.append("\n> **Cara baca:** `WR impas` = win rate minimum agar Avg R = 0 pada rasio menang/kalah yang terjadi. "
             "Strategi hanya layak jika WR (batas bawah CI95) berada di atas WR impas dan Avg R positif setelah biaya. "
             "N kecil (<100 trade closed) belum bisa dipercaya.\n")
    engines = [e for e in ("MNSR", "A+", "AMD", "ALL") if e in set(all_df["engine"])]
    L.append("\n## Hasil per engine (setelah biaya)\n")
    L.append(HEADER)
    S = {}
    for e in engines:
        S[e] = stats(all_df[all_df["engine"] == e])
        L.append(headline_row(e, S[e]))
    L.append("")
    for e in engines:
        d = all_df[all_df["engine"] == e].copy()
        if d.empty:
            continue
        d["hour_wib"] = d["t_create"].apply(lambda x: f"{datetime.fromtimestamp(x / 1000, WIB).hour:02d}")
        d["month"] = d["t_create"].apply(lambda x: datetime.fromtimestamp(x / 1000, WIB).strftime("%Y-%m"))
        d["weekday"] = d["t_create"].apply(lambda x: datetime.fromtimestamp(x / 1000, WIB).strftime("%a"))
        s = S[e]
        L.append(f"\n## Detail {e}\n")
        L.append(f"Gross R (sebelum biaya): {s['gross_R']:+.1f} | Net R: {s['total_R']:+.1f} | "
                 f"RR rencana rata-rata: {s['avg_rr_plan']:.2f} | rata-rata durasi hold: {s['avg_hold_min']:.0f} menit | "
                 f"OPEN di akhir data: {s['open_end']}\n")
        L.append("**Status:** " + ", ".join(f"{k}={v}" for k, v in d["status"].value_counts().items()) + "\n")
        L.append("**Sisi**\n\n" + group_table(d, "side") + "\n")
        L.append("**Jam WIB (waktu sinyal)**\n\n" + group_table(d.sort_values("hour_wib"), "hour_wib", sort="key") + "\n")
        L.append("**Per bulan**\n\n" + group_table(d, "month", sort="key") + "\n")
        L.append("**Hari**\n\n" + group_table(d, "weekday") + "\n")
        L.append("**10 pair terbaik (N>=5)**\n\n" + group_table(d, "symbol", min_n=5, top=10) + "\n")
        L.append("**10 pair terburuk (N>=5)**\n\n" + group_table(d, "symbol", min_n=5, top=10, sort="asc") + "\n")
        L.append("**Rentang risiko SL (% dari entry)**: "
                 f"median {d['risk_pct'].median():.2f}%, p90 {d['risk_pct'].quantile(0.9):.2f}%\n")
    L.append("\n## Simulasi portfolio (aturan bot REAL)\n")
    for e in engines:
        d = all_df[all_df["engine"] == e]
        if len(d):
            L.append(portfolio_text(e, portfolio(d, cfg), cfg))
    L.append("\n## Batasan yang harus diingat\n")
    L.append("- Entry AMD memakai candle yang baru close (live memakai candle yang sedang berjalan).\n"
             "- Filter news USD, rotasi scan 40 pair, presisi/min-notional, funding, dan likuidasi tidak dimodelkan.\n"
             "- Portfolio dihitung pasca-filter dari sinyal per-pair; sinyal yang ditolak limit tidak "
             "membebaskan slot pair tersebut, jadi hasilnya sedikit konservatif.\n"
             "- Backtest tidak menjamin hasil live; validasi tetap dengan DRY RUN forward.\n")
    return "\n".join(L), S


def equity_png(all_df, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=130)
    for e in ("MNSR", "A+", "AMD", "ALL"):
        d = all_df[(all_df["engine"] == e) & all_df["status"].isin(["TP", "SL"])].sort_values("t_exit")
        if d.empty:
            continue
        x = pd.to_datetime(d["t_exit"], unit="ms")
        ax.plot(x, d["net_R"].cumsum(), label=f"{e} (n={len(d)})", linewidth=1.4)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Equity kumulatif (R, setelah biaya)")
    ax.set_ylabel("R")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alche", default="Alche.py")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--days", type=int, default=180, help="berapa hari terakhir yang diuji")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--engines", default="MNSR,A+,AMD", help="pisahkan koma; tiap engine diuji TERPISAH")
    ap.add_argument("--combined", action="store_true", help="tambah run ALL (prioritas MNSR > A+ > AMD seperti bot)")
    ap.add_argument("--symbols", default="", help="batasi pair, mis. BTC_USDT,ETH_USDT,SOL_USDT")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--maker", type=float, default=0.0002)
    ap.add_argument("--taker", type=float, default=0.00055)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--no-price-filter", action="store_true")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--lev", type=int, default=20)
    ap.add_argument("--max-pos", type=int, default=2)
    ap.add_argument("--max-day", type=int, default=10)
    ap.add_argument("--max-loss", type=float, default=5.0)
    a = ap.parse_args()

    cfg = {"days": a.days, "maker": a.maker, "taker": a.taker, "slip": a.slip,
           "price_filter": not a.no_price_filter, "margin": a.margin, "lev": a.lev,
           "max_pos": a.max_pos, "max_day": a.max_day, "max_loss": a.max_loss}
    data_dir = Path(a.data_dir)
    files = sorted(p.name[:-7] for p in data_dir.glob("*.csv.gz"))
    if a.symbols:
        want = {s.strip() for s in a.symbols.split(",")}
        files = [f for f in files if f in want]
    if a.max_symbols:
        files = files[: a.max_symbols]
    if not files:
        sys.exit(f"Tidak ada data di {data_dir}/ (jalankan download_data.py dulu).")
    for must in ("BTC_USDT", "ETH_USDT"):
        if not (data_dir / f"{must}.csv.gz").exists():
            print(f"PERINGATAN: {must} tidak ada -> MNSR dan A+ tidak akan menghasilkan sinyal (butuh SMT BTC/ETH).")

    runs = [(e.strip(), {e.strip()}) for e in a.engines.split(",") if e.strip()]
    if a.combined:
        runs.append(("ALL", {"MNSR", "A+", "AMD"}))
    jobs = [(s, label, eng) for label, eng in runs for s in files]
    print(f"{len(files)} pair x {len(runs)} run = {len(jobs)} job, {a.days} hari, {a.workers} worker")

    all_tr = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             initargs=(a.alche, a.data_dir, cfg)) as pool:
        for (sym, trades), job in zip(pool.map(run_symbol, jobs, chunksize=1), jobs):
            for tr in trades:
                tr["engine"] = job[1] if job[1] == "ALL" else tr["engine"]
            all_tr.extend(trades)
            done += 1
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} job selesai, {len(all_tr)} sinyal")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not all_tr:
        sys.exit("Tidak ada sinyal sama sekali pada periode/data ini.")
    df = pd.DataFrame(all_tr)
    df["time_create_wib"] = df["t_create"].apply(lambda x: datetime.fromtimestamp(x / 1000, WIB).strftime("%Y-%m-%d %H:%M"))
    df.to_csv(out / "trades.csv", index=False)
    period = (f"{datetime.fromtimestamp(df['t_create'].min() / 1000, WIB):%Y-%m-%d} s/d "
              f"{datetime.fromtimestamp(df['t_exit'].max() / 1000, WIB):%Y-%m-%d} (WIB)")
    report, S = build_report(df, cfg, files, period)
    (out / "summary.md").write_text(report, encoding="utf-8")
    equity_png(df, out / "equity_R.png")
    print("\n" + HEADER)
    for e, s in S.items():
        print(headline_row(e, s))
    print(f"\nTersimpan di {out}/ (summary.md, trades.csv, equity_R.png)")


if __name__ == "__main__":
    main()
