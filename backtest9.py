#!/usr/bin/env python3
"""
Swing-Trading VPVR+ATR Walk-Forward Backtester
Refactored for modularity, type hints, and JSON-friendly output.
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import ccxt
import numpy as np
import pandas as pd
import ray
import requests
from numba import njit, prange
from requests.exceptions import HTTPError
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm


# ─── Logger Setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("backtester")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)
requests.packages.urllib3.disable_warnings()


# ─── Utilities ─────────────────────────────────────────────────────────────────
def iso_to_dt(iso: str) -> datetime:
    """Convert ISO8601 string (with Z) to timezone-aware datetime."""
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def dt_to_iso(dt: datetime) -> str:
    """Convert datetime to ISO8601 UTC string."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Data Loader ───────────────────────────────────────────────────────────────
class DataLoader:
    """Handles CoinAPI caching or CCXT fallback OHLCV retrieval."""

    def __init__(self, symbol: str, period: str, cache_dir: str, api_key: str, page_limit: int):
        self.symbol = symbol
        self.period = period
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.page_limit = page_limit
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"{symbol.replace('/', '_')}_{period}.json")

    def validate_key(self) -> None:
        """Ensure the CoinAPI key is valid (401 → exit)."""
        if not self.api_key:
            logger.error("No API key provided!")
            sys.exit(1)
        logger.info("Validating CoinAPI key…")
        try:
            resp = requests.get(
                "https://rest.coinapi.io/v1/exchanges",
                headers={"Authorization": self.api_key},
                timeout=10,
                verify=False,
            )
            resp.raise_for_status()
        except HTTPError as e:
            code = e.response.status_code if e.response else None
            msg = "invalid/expired" if code == 401 else str(e)
            logger.error(f"API key problem ({code}): {msg}")
            sys.exit(1)
        logger.info("API key OK.")

    def _fetch_coinapi(self, start: datetime, end: datetime) -> List[dict]:
        """Fetch a list of bars from CoinAPI, handling pagination."""
        url = f"https://rest.coinapi.io/v1/ohlcv/{self.symbol}/history"
        headers = {"Authorization": self.api_key, "Accept": "application/json"}
        out: List[dict] = []
        cur = start
        while cur < end:
            params = {
                "period_id": self.period,
                "time_start": dt_to_iso(cur),
                "time_end": dt_to_iso(end),
                "limit": self.page_limit,
                "sort": "ASC",
            }
            for attempt in range(3):
                resp = requests.get(url, headers=headers, params=params, timeout=30, verify=False)
                if resp.ok:
                    batch = resp.json()
                    out += batch
                    break
                time.sleep(2 ** attempt)
            else:
                logger.error("Failed to fetch from CoinAPI after retries.")
                break
            if len(batch) < self.page_limit:
                break
            last = iso_to_dt(batch[-1]["time_period_start"])
            cur = last + timedelta(seconds=1)
        return out

    def _fetch_ccxt(self, timeframe: str, start_iso: str, end_iso: str) -> pd.DataFrame:
        """Fetch OHLCV via CCXT (e.g. Binance)."""
        exchange = ccxt.binance({"enableRateLimit": True})
        since = int(iso_to_dt(start_iso).timestamp() * 1000)
        end_ts = int(iso_to_dt(end_iso).timestamp() * 1000)
        frames = []
        while since < end_ts:
            bars = exchange.fetch_ohlcv(self.symbol, timeframe, since=since, limit=500)
            if not bars:
                break
            df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            df.set_index("time", inplace=True)
            frames.append(df[df.index < pd.to_datetime(end_iso)])
            since = int(df.index[-1].timestamp() * 1000) + 1
        return pd.concat(frames)

    def _save_cache(self, bars: List[dict]) -> None:
        with open(self.cache_file, "w") as f:
            json.dump(sorted(bars, key=lambda b: b["time_period_start"]), f, indent=2)

    def _load_cache(self) -> List[dict]:
        if not os.path.exists(self.cache_file):
            return []
        with open(self.cache_file, "r") as f:
            return json.load(f)

    def ensure_cache(self, start: str, end: str) -> None:
        """Make sure cache covers [start, end), fetch missing."""
        data = self._load_cache()
        start_dt, end_dt = iso_to_dt(start), iso_to_dt(end)
        if not data:
            logger.info(f"Fetching full range {start} → {end}")
            fetched = self._fetch_coinapi(start_dt, end_dt)
            self._save_cache(fetched)
            return

        times = [iso_to_dt(b["time_period_start"]) for b in data]
        head, tail = min(times), max(times)
        to_add: List[dict] = []
        if start_dt < head:
            to_add += self._fetch_coinapi(start_dt, head - timedelta(seconds=1))
        if tail < end_dt:
            to_add += self._fetch_coinapi(tail + timedelta(seconds=1), end_dt)
        if to_add:
            merged = {b["time_period_start"]: b for b in data + to_add}
            self._save_cache(list(merged.values()))
            logger.info(f"Appended {len(to_add)} bars.")
        else:
            logger.info("Cache up to date.")

    def load_data(self, start: str, end: str, source: str) -> pd.DataFrame:
        """Return OHLCV df indexed by UTC time."""
        if source == "coinapi":
            bars = self._load_cache()
            df = (
                pd.DataFrame(bars)
                .rename(columns={
                    "time_period_start": "time",
                    "price_open": "open",
                    "price_high": "high",
                    "price_low": "low",
                    "price_close": "close",
                    "volume_traded": "volume",
                })
                .assign(time=lambda x: pd.to_datetime(x["time"], utc=True))
                .set_index("time")
                .sort_index()
            )
            mask = (df.index >= iso_to_dt(start)) & (df.index < iso_to_dt(end))
            return df.loc[mask, ["open", "high", "low", "close", "volume"]]
        else:
            return self._fetch_ccxt(self.period, start, end)


# ─── Indicators ────────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def calc_tr_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    atr = np.empty(n, dtype=np.float64)
    atr[0] = high[0] - low[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        tr = max(high[i] - low[i],
                 abs(high[i] - close[i - 1]),
                 abs(low[i] - close[i - 1]))
        atr[i] = alpha * tr + (1 - alpha) * atr[i - 1]
    return atr

@njit(cache=True, fastmath=True, parallel=True)
def calc_vpvr_numba_optimized(
    low_arr: np.ndarray, high_arr: np.ndarray, vol_arr: np.ndarray,
    num_bins: int, low_val: float, high_val: float
) -> Tuple[np.ndarray, float]:
    height = (high_val - low_val) / num_bins
    inv_h = 1.0 / height
    vol_hist = np.zeros(num_bins, dtype=np.float64)
    for i in prange(len(low_arr)):
        l, h, v = low_arr[i], high_arr[i], vol_arr[i]
        if h <= low_val or l >= high_val or h - l <= 0:
            continue
        cl = max(l, low_val); ch = min(h, high_val)
        start = int((cl - low_val) * inv_h); end = int((ch - low_val) * inv_h)
        bar_h = h - l; factor = v / bar_h
        for b in range(max(0, start), min(num_bins, end + 1)):
            bin_low = low_val + b * height
            bin_high = bin_low + height
            overlap = min(bin_high, ch) - max(bin_low, cl)
            if overlap > 0:
                vol_hist[b] += factor * overlap
    return vol_hist, height


# ─── Simulator ─────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    dir: str
    entry: float
    exit: float
    pnl: float
    size: float

def simulate_vpvr(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Run one pass of the VPVR+ATR strategy on df with cfg.
    Returns DataFrame of trades.
    """
    o, h, l, c, vol = (
        df["open"].values, df["high"].values,
        df["low"].values, df["close"].values,
        df["volume"].values,
    )
    times = df.index.values
    ema_fast = df["close"].ewm(span=cfg["ma_fast"], adjust=False).mean().values
    ema_slow = df["close"].ewm(span=cfg["ma_slow"], adjust=False).mean().values
    atr_arr = calc_tr_atr_numba(h, l, c, cfg["atr_period"])

    trades: List[Trade] = []
    equity = cfg["init_eq"]
    position = None
    entry_idx = entry_px = target_px = None
    slippage_entry = 0.0

    for i in tqdm(range(cfg["warmup"], len(df) - 1),
                  desc="Simulating VPVR", unit="bar"):
        prev_c = c[i - 1]
        # ENTRY logic…
        if position is None:
            window_start = max(0, i - cfg["vpvr_window"] + 1)
            lows, highs, vols = l[window_start : i+1], h[window_start : i+1], vol[window_start : i+1]
            vmin, vmax = float(lows.min()), float(highs.max())
            profile, height = calc_vpvr_numba_optimized(
                lows, highs, vols, cfg["num_bins"], vmin, vmax
            )
            peaks = sorted([vmin + (b + 0.5)*height for b in
                            np.argpartition(profile, -cfg["top_n_peaks"])[-cfg["top_n_peaks"]:]], reverse=True)
            valleys = sorted([vmin + (b + 0.5)*height for b in
                              np.argpartition(profile, cfg["top_n_valleys"])[:cfg["top_n_valleys"]]])
            buy_lvls = [v for v in valleys if v > prev_c]
            sell_lvls = [p for p in peaks if p < prev_c]
            buy_trig = (min(buy_lvls)*(1+cfg["entry_spread"])
                        if buy_lvls else None)
            sell_trig = (max(sell_lvls)*(1-cfg["entry_spread"])
                         if sell_lvls else None)

            if buy_trig and h[i] >= buy_trig:
                position, entry_idx = "long", i + 1
                raw_px = o[entry_idx]*(1+cfg["entry_spread"])
            elif sell_trig and l[i] <= sell_trig:
                position, entry_idx = "short", i + 1
                raw_px = o[entry_idx]*(1-cfg["entry_spread"])
            else:
                raw_px = None

            if position:
                # Size & slippage
                if cfg["dynamic_size"]:
                    stop_price = (raw_px - atr_arr[entry_idx]*cfg["stop_atr_mult"]
                                  if position=="long"
                                  else raw_px + atr_arr[entry_idx]*cfg["stop_atr_mult"])
                    risk = abs(raw_px - stop_price)
                    size = (equity*cfg["risk_pct"]) / risk
                else:
                    size = equity*cfg["risk_pct"]*cfg["leverage"]
                slip_px = raw_px*(1+cfg["slippage_pct"] if position=="long"
                                 else 1-cfg["slippage_pct"])
                slippage_entry = abs(slip_px - raw_px)*size
                entry_px = slip_px
                # target price
                if position=="long":
                    higher = [p for p in peaks if p > buy_trig/(1+cfg["entry_spread"])]
                    target_px = higher[0] if higher else entry_px*(1+cfg["min_tp_sl_distance"])
                else:
                    lower = [v for v in valleys if v < sell_trig/(1-cfg["entry_spread"])]
                    target_px = lower[-1] if lower else entry_px*(1-cfg["min_tp_sl_distance"])

        # EXIT logic…
        if position:
            exit_px: Optional[float] = None
            if position=="long" and h[i] >= target_px:
                exit_px = target_px
            elif position=="short" and l[i] <= target_px:
                exit_px = target_px

            # trend stop
            if exit_px is None and i >= cfg["trend_lookback"]:
                xs = np.arange(i-cfg["trend_lookback"], i)
                ys = c[i-cfg["trend_lookback"]:i]
                m, b = np.polyfit(xs, ys, 1)
                trend_val = m*i + b
                if position=="long" and l[i] < trend_val:
                    exit_px = trend_val
                if position=="short" and h[i] > trend_val:
                    exit_px = trend_val

            # EMA cross
            if exit_px is None and position=="long" and ema_fast[i] < ema_slow[i]:
                exit_px = o[i+1]
            elif exit_px is None and position=="short" and ema_fast[i] > ema_slow[i]:
                exit_px = o[i+1]

            # max hold
            if exit_px is None and (i - entry_idx) >= cfg["max_hold_period"]:
                exit_px = o[i+1]

            if exit_px is not None:
                slip_exit = exit_px*(1-cfg["slippage_pct"]
                                     if position=="long"
                                     else 1+cfg["slippage_pct"])
                slippage_exit = abs(slip_exit - exit_px)*size
                eff_exit = slip_exit
                commission = cfg["commission"] * size * (entry_px + eff_exit)
                pnl = ((1 if position=="long" else -1)*(eff_exit-entry_px)*size
                       - commission - slippage_entry - slippage_exit)
                trades.append(Trade(
                    entry_time=pd.Timestamp(times[entry_idx]),
                    exit_time=pd.Timestamp(times[i+1]),
                    dir=position,
                    entry=entry_px,
                    exit=eff_exit,
                    pnl=pnl,
                    size=size
                ))
                equity += pnl
                position = None

    if not trades:
        return pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
    return pd.DataFrame([t.__dict__ for t in trades])


# ─── Performance ───────────────────────────────────────────────────────────────
def calculate_performance(trades: pd.DataFrame, init_eq: float) -> Dict[str, Any]:
    """Compute net PnL, equity curve, win rate, PF, drawdown, Sharpe."""
    if trades.empty:
        return {"net": 0.0, "equity_curve": pd.DataFrame(), "win_rate": np.nan,
                "avg_win": np.nan, "avg_loss": np.nan, "profit_factor": np.nan,
                "max_drawdown": np.nan, "sharpe": np.nan}
    eq = init_eq
    curve: List[Tuple[pd.Timestamp, float]] = []
    rets: List[float] = []
    for _, row in trades.sort_values("exit_time").iterrows():
        ret = row["pnl"] / eq
        rets.append(ret); eq += row["pnl"]
        curve.append((row["exit_time"], eq))
    df_curve = pd.DataFrame(curve, columns=["time", "equity"])
    net = eq - init_eq
    wins = trades["pnl"] > 0
    gross_w = trades.loc[wins, "pnl"].sum()
    gross_l = -trades.loc[~wins, "pnl"].sum()
    return {
        "net": net,
        "equity_curve": df_curve,
        "win_rate": wins.mean(),
        "avg_win": trades.loc[wins, "pnl"].mean() if wins.any() else 0.0,
        "avg_loss": trades.loc[~wins, "pnl"].mean() if (~wins).any() else 0.0,
        "profit_factor": gross_w / gross_l if gross_l > 0 else np.inf,
        "max_drawdown": ((df_curve.equity - df_curve.equity.cummax()) / df_curve.equity.cummax()).min(),
        "sharpe": (np.mean(rets) / np.std(rets) * np.sqrt(len(rets))) if len(rets) > 1 else np.nan
    }


# ─── Optimizer ─────────────────────────────────────────────────────────────────
@ray.remote(num_cpus=1)
def _eval_params(params: Dict[str, Any], df: pd.DataFrame, base: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
    """Remote evaluation for one hyperparameter set."""
    cfg = {**base, **params}
    tscv = TimeSeriesSplit(n_splits=base["n_splits"])
    scores: List[float] = []
    for train_idx, val_idx in tscv.split(df):
        perf_tr = calculate_performance(simulate_vpvr(df.iloc[train_idx], cfg), cfg["init_eq"])
        if perf_tr["net"] <= 0:
            return None
        perf_va = calculate_performance(simulate_vpvr(df.iloc[val_idx], cfg), cfg["init_eq"])
        if perf_va["equity_curve"].empty:
            return None
        scores.append(perf_va["net"])
    return float(np.mean(scores)), params

def optimize_with_ray(df: pd.DataFrame, space: Dict[str, Dict[str,int]],
                      base: Dict[str, Any], n_trials: int, top_n: int
) -> Tuple[List[Tuple[float, Dict[str,Any]]], List[Tuple[float, Dict[str,Any]]]]:
    """Run random hyperparameter search with Ray and return all & top_n results."""
    if not ray.is_initialized():
        ray.init(include_dashboard=False)
    trials = []
    for _ in range(n_trials):
        p = {k: int(np.random.randint(v["min"], v["max"] + 1)) for k, v in space.items()}
        trials.append(p)
    df_id = ray.put(df)
    base_id = ray.put(base)
    futures = [_eval_params.remote(p, df_id, base_id) for p in trials]

    results: List[Tuple[float, Dict[str,Any]]] = []
    for f in tqdm(futures, desc="Eval Ray", unit="trial"):
        res = ray.get(f)
        if res is not None:
            results.append(res)
    results.sort(key=lambda x: x[0], reverse=True)
    return results, results[:top_n]


# ─── Plotting & JSON Output ────────────────────────────────────────────────────
def plot_equity(curve: pd.DataFrame, title: str) -> None:
    """Simple equity curve line chart."""
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(curve["time"], curve["equity"])
    plt.title(title)
    plt.xlabel("Time"); plt.ylabel("Equity")
    plt.tight_layout()


# ─── Main Entrypoint ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser("VPVR+ATR Swing Backtester")
    p.add_argument("--symbol", required=True)
    p.add_argument("--period", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--forward_end", default=None)
    p.add_argument("--api_key", default=os.getenv("COINAPI_KEY"))
    p.add_argument("--page_limit", type=int, default=100)
    p.add_argument("--n_trials", type=int, default=200)
    p.add_argument("--top_n", type=int, default=5)
    p.add_argument("--data_source", choices=["coinapi","ccxt"], default="coinapi")
    p.add_argument("--llm_output", action="store_true",
                   help="Print final metrics as JSON and exit.")
    args = p.parse_args()

    loader = DataLoader(args.symbol, args.period,
                        cache_dir=os.path.join(os.getcwd(), args.symbol.replace("/", "_")),
                        api_key=args.api_key, page_limit=args.page_limit)
    if args.data_source == "coinapi":
        loader.validate_key()
        loader.ensure_cache(args.start, args.end)
    df_in = loader.load_data(args.start, args.end, args.data_source)

    cfg_base = dict(
        num_bins=50, warmup=200, max_hold_period=48,
        commission=0.0004, leverage=10, risk_pct=0.01,
        init_eq=100_000, n_splits=5, min_tp_sl_distance=0.003,
        entry_spread=0.0005, vpvr_window=200,
        top_n_peaks=3, top_n_valleys=3,
        atr_period=14, ma_fast=20, ma_slow=50,
        trend_lookback=100, slippage_pct=0.0005,
        dynamic_size=True, stop_atr_mult=2.0
    )
    space = dict(
        vpvr_window={"min":50,"max":500},
        top_n_peaks={"min":1,"max":10},
        top_n_valleys={"min":1,"max":10},
        atr_period={"min":5,"max":50},
        ma_fast={"min":10,"max":200},
        ma_slow={"min":50,"max":500},
        max_hold_period={"min":24,"max":240},
        trend_lookback={"min":20,"max":200},
    )

    logger.info("=== In-Sample Optimization ===")
    all_results, best = optimize_with_ray(df_in, space, cfg_base, args.n_trials, args.top_n)
    if not best:
        logger.error("No profitable in-sample configs found."); sys.exit(1)

    # Collect metrics
    summary: List[Dict[str, Any]] = []
    for rank, (score, params) in enumerate(best, start=1):
        cfg = {**cfg_base, **params}
        trades = simulate_vpvr(df_in, cfg)
        perf = calculate_performance(trades, cfg["init_eq"])
        summary.append({"rank": rank, "score": score, **params, **perf})
        logger.info(f"[In-Sample #{rank}] PnL={perf['net']:.2f}, WinRate={perf['win_rate']:.2%}, "
                    f"Sharpe={perf['sharpe']:.2f}, MaxDD={perf['max_drawdown']:.2%}")
        if not args.llm_output:
            plot_equity(perf["equity_curve"], f"In-Sample Equity #{rank}")

    # Forward test
    fwd_end = args.forward_end or dt_to_iso(datetime.now(timezone.utc))
    logger.info(f"=== Forward Test ({args.end} → {fwd_end}) ===")
    if args.data_source == "coinapi":
        loader.ensure_cache(args.end, fwd_end)
    df_fw = loader.load_data(args.end, fwd_end, args.data_source)

    for rank, entry in enumerate(best, start=1):
        _, params = entry
        cfg = {**cfg_base, **params}
        trades = simulate_vpvr(df_fw, cfg)
        perf = calculate_performance(trades, cfg["init_eq"])
        summary[rank-1].update({f"fwd_{k}": v for k, v in perf.items() if k != "equity_curve"})
        logger.info(f"[Forward #{rank}] PnL={perf['net']:.2f}, WinRate={perf['win_rate']:.2%}, "
                    f"Sharpe={perf['sharpe']:.2f}, MaxDD={perf['max_drawdown']:.2%}")
        if not args.llm_output:
            plot_equity(perf["equity_curve"], f"Forward Equity #{rank}")

    if args.llm_output:
        print(json.dumps(summary, default=lambda o: o if not isinstance(o, pd.DataFrame) else o.to_dict("list")))
    else:
        import matplotlib.pyplot as plt
        logger.info("Done – displaying plots.")
        plt.show()


if __name__ == "__main__":
    main()
