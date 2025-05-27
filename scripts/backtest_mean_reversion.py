#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

# Data paths
DATA_DIR = Path(project_root) / "data" / "historical" / "csv"
SYMBOL = "26000"  # NIFTY

# ------------ Load Data ------------
daily = pd.read_csv(DATA_DIR / "1day" / f"{SYMBOL}.csv", parse_dates=["timestamp"])
five = pd.read_csv(DATA_DIR / "5min" / f"{SYMBOL}.csv", parse_dates=["timestamp"])
fifteen = pd.read_csv(DATA_DIR / "15min" / f"{SYMBOL}.csv", parse_dates=["timestamp"])

print("\nData loaded:")
print(f"Daily data: {len(daily)} rows from {daily.timestamp.min()} to {daily.timestamp.max()}")
print(f"5min data: {len(five)} rows from {five.timestamp.min()} to {five.timestamp.max()}")
print(f"15min data: {len(fifteen)} rows from {fifteen.timestamp.min()} to {fifteen.timestamp.max()}")

# Drop timezone and index
for df in (daily, five, fifteen):
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

daily.set_index("timestamp", inplace=True)
five.set_index("timestamp", inplace=True)
fifteen.set_index("timestamp", inplace=True)

# ------------ Generate Daily Signals ------------
daily["returns"] = daily["close"].pct_change()
window = 20
daily["mu"] = daily["returns"].rolling(window).mean()
daily["sigma"] = daily["returns"].rolling(window).std()
daily["z"] = (daily["returns"] - daily["mu"]) / daily["sigma"]
signals = daily[(daily["z"] >= 1.5) | (daily["z"] <= -1.5)].copy()
signals["Signal"] = np.where(signals["z"] <= -1.5, "Long", "Short")
signals = signals.reset_index()[["timestamp", "Signal"]]
signals.rename(columns={"timestamp": "Date"}, inplace=True)

# Limit to backtest window - using last 3 months of data
end_date = pd.Timestamp(daily.index[-1].date())
start_date = end_date - pd.DateOffset(months=3)
signals = signals[(signals["Date"] >= start_date) & (signals["Date"] <= end_date)]

print(f"\nBacktest period: {start_date.date()} to {end_date.date()}")
print(f"Total signals generated: {len(signals)}")

# ------------ Prepare intraday indices ------------
five_idx = five.index
fifteen_idx = fifteen.index
max_date = fifteen_idx[-1].date()

SL_PCT = 0.01   # 1%
TP_PCT = 0.015  # 1.5%

trade_log = []

for _, row in signals.iterrows():
    sig_date = row["Date"]
    side     = row["Signal"]
    
    # find next trading day's 09:20 bar in 5-min
    day = sig_date + timedelta(days=1)
    entry_ts = None
    while day.date() <= max_date:
        if day.weekday() < 5:
            ts = pd.Timestamp.combine(day.date(), pd.Timestamp("09:20:00").time())
            if ts in five_idx:
                entry_ts = ts
                break
        day += timedelta(days=1)
    if entry_ts is None:
        continue
    
    entry_px = five.at[entry_ts, "open"]
    sl_px = entry_px * (1 - SL_PCT) if side == "Long" else entry_px * (1 + SL_PCT)
    tp_px = entry_px * (1 + TP_PCT) if side == "Long" else entry_px * (1 - TP_PCT)
    
    day_slice = fifteen.loc[
        pd.Timestamp.combine(entry_ts.date(), pd.Timestamp("09:15:00").time()):
        pd.Timestamp.combine(entry_ts.date(), pd.Timestamp("15:15:00").time())
    ]
    
    exit_ts, exit_px = None, None
    for ts, bar in day_slice.iterrows():
        hi, lo = bar["high"], bar["low"]
        if side == "Long":
            if lo <= sl_px:
                exit_ts, exit_px = ts, sl_px
                break
            if hi >= tp_px:
                exit_ts, exit_px = ts, tp_px
                break
        else:
            if hi >= sl_px:
                exit_ts, exit_px = ts, sl_px
                break
            if lo <= tp_px:
                exit_ts, exit_px = ts, tp_px
                break
    if exit_ts is None:
        exit_ts = day_slice.index[-1]
        exit_px = day_slice.iloc[-1]["close"]
    
    pnl_pts = (exit_px - entry_px) if side == "Long" else (entry_px - exit_px)
    outcome = "Win" if pnl_pts > 0 else "Loss"
    
    trade_log.append({
        "Entry Time": entry_ts,
        "Side": side,
        "Entry Px": round(entry_px,2),
        "Exit Time": exit_ts,
        "Exit Px": round(exit_px,2),
        "PnL (pts)": round(pnl_pts,2),
        "Outcome": outcome
    })

trades = pd.DataFrame(trade_log)

# ------------ Metrics ------------
total = len(trades)
wins  = trades[trades["Outcome"]=="Win"]
losses= trades[trades["Outcome"]=="Loss"]

summary = pd.DataFrame([{
    "Total Trades": total,
    "Wins": len(wins),
    "Win Rate %": round(len(wins)/total*100,2) if total else 0,
    "Net P/L (pts)": round(trades["PnL (pts)"].sum(),2),
    "Avg Win (pts)": round(wins["PnL (pts)"].mean(),2) if not wins.empty else 0,
    "Avg Loss (pts)": round(losses["PnL (pts)"].mean(),2) if not losses.empty else 0
}])

print("\nBack-test Summary (1% SL / 1.5% TP)")
print("=" * 50)
print(summary.to_string(index=False))
print("\nTrades Log (first 10)")
print("=" * 50)
print(trades.head(10).to_string())

# Save results
results_dir = Path(project_root) / "results" / "mean_reversion"
results_dir.mkdir(parents=True, exist_ok=True)

trades.to_csv(results_dir / "trades.csv", index=False)
summary.to_csv(results_dir / "summary.csv", index=False)
print(f"\nResults saved to: {results_dir}") 