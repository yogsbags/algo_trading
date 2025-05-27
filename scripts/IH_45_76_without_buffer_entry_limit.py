def breach_low_no_buffer_time_filtered(df, SL=45.0, TP=76.0, cutoff_time=datetime.time(13, 45)):
    df = df.copy()
    candle_range = df['high'] - df['low']
    body = (df['open'] - df['close']).abs()

    df['iham'] = (
        (candle_range > 3 * body) &
        ((df['high'] - df['close']) / candle_range.replace(0, 1e-9) > 0.6) &
        ((df['high'] - df['open']) / candle_range.replace(0, 1e-9) > 0.6)
    )

    df['entry_signal'] = df['iham'].shift(1) & (df['low'] < df['low'].shift(1))
    prev_low = df['low'].shift(1)

    trades = []
    in_pos = False
    for ts, row in df.iterrows():
        if not in_pos and row['entry_signal'] and ts.time() < cutoff_time:
            entry_price = prev_low.loc[ts] if not np.isnan(prev_low.loc[ts]) else row['open']
            sl, tp = entry_price + SL, entry_price - TP
            entry_time = ts
            in_pos = True
            continue

        if in_pos:
            high, low = row['high'], row['low']
            exit_price = None
            if high >= sl and low <= tp:
                exit_price, reason = sl, 'SL'  # pessimistic
            elif low <= tp:
                exit_price, reason = tp, 'TP'
            elif high >= sl:
                exit_price, reason = sl, 'SL'
            elif ts.time() >= datetime.time(15, 15):
                exit_price, reason = row['close'], 'EOD'

            if exit_price is not None:
                pnl = entry_price - exit_price
                trades.append({
                    'Entry Time': entry_time, 'Exit Time': ts,
                    'Entry Price': round(entry_price, 2),
                    'Exit Price': round(exit_price, 2),
                    'Reason': reason,
                    'PnL (pts)': round(pnl, 2)
                })
                in_pos = False
    return pd.DataFrame(trades)
