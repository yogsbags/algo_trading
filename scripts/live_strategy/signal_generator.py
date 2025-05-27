import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Signal generation logic for straddle strategy"""
    
    def __init__(self, 
                 sl_long: int = 70,
                 tp_long: int = 100,
                 sl_short: int = 60,
                 tp_short: int = 90,
                 activation_gap: float = 100.0,
                 trail_offset: float = 50.0):
        """Initialize signal generator with strategy parameters"""
        self.sl_long = sl_long
        self.tp_long = tp_long
        self.sl_short = sl_short
        self.tp_short = tp_short
        self.activation_gap = activation_gap
        self.trail_offset = trail_offset

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on price action and VWAP"""
        logger.info(f"[DEBUG] generate_signals called with {len(df)} rows. Columns: {df.columns.tolist()}")
        if len(df) == 0:
            logger.info("[DEBUG] DataFrame is empty, returning early.")
            return df.copy()
        
        df = df.copy()  # Always work on a copy to avoid SettingWithCopyWarning
        df.loc[:, 'signal'] = None
        
        # Ensure timestamp is datetime before using .dt
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            logger.warning("[SignalGenerator] Converting 'timestamp' column to datetime for .dt accessor safety.")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for day, day_data in df.groupby(df.timestamp.dt.date):
            logger.info(f"[DEBUG] Processing day: {day} with {len(day_data)} rows")
            day_df = day_data.sort_values('timestamp').reset_index(drop=True)
            
            if len(day_df) < 3:
                logger.info(f"[DEBUG] Not enough data for day {day} (len={len(day_df)}), skipping pattern logic.")
                continue
            
            # Initialize highest_high to start price so first HH is only logged if price exceeds start
            highest_high = day_df.iloc[0].straddle_price
            highest_low = None
            # Initialize lowest_low to start price so first LL is only logged if price drops below start
            lowest_low = day_df.iloc[0].straddle_price
            lowest_high = None
            hh_hl_sequence_formed = False
            ll_lh_sequence_formed = False
            # --- Armed LL-LH and HH-HL sequence logic ---
            ll_lh_sequence_armed = False
            armed_ll_price = None
            hh_hl_sequence_armed = False
            armed_hh_price = None
            last_pattern_type = None
            has_hh_occurred = False  # Track if a Higher High has occurred this day
            has_ll_occurred = False  # Track if a Lower Low has occurred this day
            # --- Long1/Long2/Short state ---
            in_long1 = False
            in_long2 = False
            in_short = False
            long1_entry_price = None
            long2_entry_price = None
            long1_exit_price = None
            short_entry_price = None
            trail_tp = None
            logger.info(f"[DEBUG] Start price: {day_df.iloc[0].straddle_price}, Start vwap: {getattr(day_df.iloc[0], 'vwap', getattr(day_df.iloc[0], 'vwap_5min', None))}, Timestamp: {day_df.iloc[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            # Log every candle's price and VWAP for the day
            for idx, row in day_df.iterrows():
                ts_str = row.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                call_open = getattr(row, 'call_open', None)
                put_open = getattr(row, 'put_open', None)
                vwap_val = getattr(row, 'vwap', getattr(row, 'vwap_5min', None))
                # Format values safely, avoiding errors if None or NaN
                call_open_str = f"{call_open:.2f}" if pd.notna(call_open) else "NA"
                put_open_str = f"{put_open:.2f}" if pd.notna(put_open) else "NA"
                vwap_str = f"{vwap_val:.2f}" if pd.notna(vwap_val) else "NA"
                logger.info(f"[CANDLE] {ts_str}: Call_open: {call_open_str}, Put_open: {put_open_str}, Straddle: {row.straddle_price:.2f}, VWAP: {vwap_str}")

            for i in range(2, len(day_df)):
                r, p1, p2 = day_df.iloc[i], day_df.iloc[i-1], day_df.iloc[i-2]
                # Existing pattern logic and [CANDLE] log for pattern context
                # --- EOD Exit for all trades ---
                # --- Force all trades to exit exactly at 15:20 ---
                eod_time = r.timestamp.replace(hour=15, minute=20, second=0, microsecond=0)
                if r.timestamp.time() > pd.to_datetime('15:20').time() and (
                    in_short or in_long1 or in_long2):
                    # Find previous and next candle surrounding 15:20
                    prev_idx = i - 1
                    prev_row = day_df.iloc[prev_idx]
                    prev_time = prev_row.timestamp
                    prev_price = prev_row.straddle_price
                    next_time = r.timestamp
                    next_price = r.straddle_price
                    # Linear interpolation for price at 15:20
                    total_seconds = (next_time - prev_time).total_seconds()
                    if total_seconds == 0:
                        interp_price = next_price
                    else:
                        seconds_to_eod = (eod_time - prev_time).total_seconds()
                        interp_price = prev_price + (next_price - prev_price) * (seconds_to_eod / total_seconds)
                    # Exit all trades at 15:20 with interpolated price
                    if in_short and short_entry_price is not None:
                        logger.info(f"[SignalGenerator] Short exited at 15:20 entry {short_entry_price} exit {interp_price:.2f} (EOD), P&L: {short_entry_price - interp_price:.2f}")
                        # Update DataFrame for Short EOD exit
                        if 'exit_time' not in df.columns:
                            df['exit_time'] = None
                        if 'exit_price' not in df.columns:
                            df['exit_price'] = None
                        # Find the last Short entry (open position)
                        last_short_idx = df[(df['signal'] == 'Short') & (df['exit_time'].isnull())].index
                        if len(last_short_idx) > 0:
                            df.loc[last_short_idx, 'exit_time'] = eod_time
                            df.loc[last_short_idx, 'exit_price'] = interp_price
                        in_short = False
                        short_entry_price = None
                    if in_long1 and long1_entry_price is not None:
                        logger.info(f"[SignalGenerator] Long1 exited at 15:20 entry {long1_entry_price} exit {interp_price:.2f} (EOD), P&L: {interp_price - long1_entry_price:.2f}")
                        # Update DataFrame for Long1 EOD exit
                        if 'exit_time' not in df.columns:
                            df['exit_time'] = None
                        if 'exit_price' not in df.columns:
                            df['exit_price'] = None
                        last_long1_idx = df[(df['signal'].str.contains('Long1', na=False)) & (df['exit_time'].isnull())].index
                        if len(last_long1_idx) > 0:
                            df.loc[last_long1_idx, 'exit_time'] = eod_time
                            df.loc[last_long1_idx, 'exit_price'] = interp_price
                        in_long1 = False
                        long1_exit_price = interp_price
                    if in_long2 and long2_entry_price is not None:
                        logger.info(f"[SignalGenerator] Long2 exited at 15:20 entry {long2_entry_price} exit {interp_price:.2f} (EOD), P&L: {interp_price - long2_entry_price:.2f}")
                        # Update DataFrame for Long2 EOD exit
                        if 'exit_time' not in df.columns:
                            df['exit_time'] = None
                        if 'exit_price' not in df.columns:
                            df['exit_price'] = None
                        last_long2_idx = df[(df['signal'].str.contains('Long2', na=False)) & (df['exit_time'].isnull())].index
                        if len(last_long2_idx) > 0:
                            df.loc[last_long2_idx, 'exit_time'] = eod_time
                            df.loc[last_long2_idx, 'exit_price'] = interp_price
                        in_long2 = False
                    break  # Exit loop after EOD

                logger.debug(f"[DEBUG] i={i}, r.time={r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}, r.straddle_price={r.straddle_price}, r.vwap={getattr(r, 'vwap', getattr(r, 'vwap_5min', None))}")
                if pd.isna(getattr(r, 'vwap', getattr(r, 'vwap_5min', None))):
                    logger.debug(f"[DEBUG] Skipping i={i} due to NaN VWAP.")
                    continue
                hh1 = max(p1.straddle_price, p2.straddle_price)
                hl1 = min(p1.straddle_price, p2.straddle_price)
                ll1 = min(p1.straddle_price, p2.straddle_price)
                lh1 = max(p1.straddle_price, p2.straddle_price)
                if r.timestamp.time() == pd.to_datetime('09:15').time():
                    logger.debug(f"[DEBUG] Skipping VWAP crossing check for first candle at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}.")
                    continue
                # VWAP crossing checks
                if (p1.straddle_price < getattr(p1, 'vwap', getattr(p1, 'vwap_5min', None)) and r.straddle_price > getattr(r, 'vwap', getattr(r, 'vwap_5min', None))):
                    logger.info(f"[DEBUG] Straddle crossed above VWAP at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, VWAP: {getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                elif (p1.straddle_price > getattr(p1, 'vwap', getattr(p1, 'vwap_5min', None)) and r.straddle_price < getattr(r, 'vwap', getattr(r, 'vwap_5min', None))):
                    logger.info(f"[DEBUG] Straddle crossed below VWAP at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, VWAP: {getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                # Pattern logic (HH/HL, LL/LH, etc.)
                # Add debug logs for each pattern branch
                if r.straddle_price > getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):
                    logger.debug(f"[DEBUG] Above VWAP: r.straddle_price={r.straddle_price:.2f}, r.vwap={getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                elif r.straddle_price < getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):
                    logger.debug(f"[DEBUG] Below VWAP: r.straddle_price={r.straddle_price:.2f}, r.vwap={getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                
                # New Higher High
                current_vwap = getattr(r, 'vwap', getattr(r, 'vwap_5min', None))
                if current_vwap is None:
                    logger.warning(f"[Pattern] VWAP is None at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}, skipping pattern check")
                    continue
                    
                if (highest_high is None or r.straddle_price > highest_high) and r.straddle_price > current_vwap:
                    highest_high = r.straddle_price
                    logger.info(f"[Pattern] New Higher High (HH) at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, VWAP: {current_vwap:.2f}")
                    if last_pattern_type == 'HL':
                        # Arm the HH-HL sequence for a potential Long
                        hh_hl_sequence_armed = True
                        armed_hh_price = highest_high
                        logger.info(f"[Pattern] HH-HL sequence formed at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, VWAP: {current_vwap:.2f}")
                        hl_debug = f", HL: {highest_low:.2f}" if highest_low is not None else ""
                        logger.info(f"[Pattern] HH: {highest_high:.2f}{hl_debug}")
                        logger.debug(f"[DEBUG] HH-HL sequence armed with HH={armed_hh_price}")
                    last_pattern_type = 'HH'
                    has_hh_occurred = True  # Mark that a HH has occurred

                # --- Armed HH-HL Long trigger ---
                if hh_hl_sequence_armed and not in_long1 and not in_long2 and not in_short:
                    current_vwap = getattr(r, 'vwap', getattr(r, 'vwap_5min', None))
                    try:
                        if current_vwap is None:
                            logger.warning(f"[Pattern] VWAP is None, cannot check HH-HL trigger at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        elif r.straddle_price > armed_hh_price and r.straddle_price > current_vwap:
                            logger.info(f"[Pattern] Armed HH-HL triggered Long1 at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                      f"Price: {r.straddle_price:.2f}, Armed_HH: {armed_hh_price:.2f}, VWAP: {current_vwap:.2f}")
                            if df.loc[df['timestamp'] == r.timestamp, 'signal'].isnull().all():
                                df.loc[df['timestamp'] == r.timestamp, 'signal'] = 'Long1,Long2'
                                in_long1 = True
                                in_long2 = True
                                long1_entry_price = r.straddle_price
                                long2_entry_price = r.straddle_price
                                trail_tp = r.straddle_price + self.tp_long + self.trail_offset
                                logger.info(f"[SignalGenerator] Signal generated: Long1,Long2 at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} price {r.straddle_price:.2f}")
                                hl_debug = f", Previous HL: {highest_low:.2f}" if highest_low is not None else ""
                                logger.info(f"[Pattern] Previous HH: {armed_hh_price:.2f}{hl_debug}")
                            hh_hl_sequence_armed = False
                        else:
                            logger.debug(f"[DEBUG] Armed HH-HL not triggered: straddle_price={r.straddle_price:.2f}, "
                                      f"armed_hh_price={armed_hh_price:.2f}, vwap={current_vwap:.2f}")
                    except Exception as e:
                        logger.error(f"[ERROR] Error in HH-HL trigger: {str(e)}", exc_info=True)

                # --- Armed LL-LH Short trigger ---
                if ll_lh_sequence_armed and not in_short and not in_long1 and not in_long2:
                    current_vwap = getattr(r, 'vwap', getattr(r, 'vwap_5min', None))
                    try:
                        if current_vwap is None:
                            logger.warning(f"[Pattern] VWAP is None, cannot check LL-LH trigger at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        elif r.straddle_price < armed_ll_price and r.straddle_price < current_vwap:
                            logger.info(f"[Pattern] Armed LL-LH triggered Short at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                      f"Price: {r.straddle_price:.2f}, Armed_LL: {armed_ll_price:.2f}, VWAP: {current_vwap:.2f}")
                            df.loc[df['timestamp'] == r.timestamp, 'signal'] = 'Short'
                            in_short = True
                            short_entry_price = r.straddle_price
                            logger.info(f"[SignalGenerator] Signal generated: Short at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} price {r.straddle_price:.2f}")
                            ll_lh_sequence_armed = False
                        else:
                            logger.debug(f"[DEBUG] Armed LL-LH not triggered: straddle_price={r.straddle_price:.2f}, "
                                      f"armed_ll_price={armed_ll_price:.2f}, vwap={current_vwap:.2f}")
                    except Exception as e:
                        logger.error(f"[ERROR] Error in LL-LH trigger: {str(e)}", exc_info=True)

                # New Higher Low
                elif has_hh_occurred and highest_high is not None and r.straddle_price < highest_high and (highest_low is None or r.straddle_price > highest_low):
                    current_vwap = getattr(r, 'vwap', getattr(r, 'vwap_5min', None))
                    try:
                        if current_vwap is not None and r.straddle_price > current_vwap:
                            highest_low = r.straddle_price
                            logger.info(f"[Pattern] New Higher Low (HL) at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                      f"Price: {r.straddle_price:.2f}, VWAP: {current_vwap:.2f}")
                            if last_pattern_type == 'HH':
                                hh_hl_sequence_formed = True
                                logger.info(f"[Pattern] HH-HL sequence formed at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                          f"Price: {r.straddle_price:.2f}, VWAP: {current_vwap:.2f}")
                                hh_debug = f", HL: {highest_low:.2f}" if highest_low is not None else ""
                                logger.info(f"[Pattern] HH: {highest_high:.2f}{hh_debug}")
                            last_pattern_type = 'HL'
                    except Exception as e:
                        logger.error(f"[ERROR] Error in Higher Low detection: {str(e)}", exc_info=True)
                # New Lower Low
                elif lowest_low is None or r.straddle_price < lowest_low:
                    current_vwap = getattr(r, 'vwap', getattr(r, 'vwap_5min', None))
                    try:
                        if current_vwap is not None and r.straddle_price < current_vwap:
                            prev_lowest_low = lowest_low
                            lowest_low = r.straddle_price
                            logger.info(f"[Pattern] New Lower Low (LL) at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                      f"Price: {r.straddle_price:.2f}, VWAP: {current_vwap:.2f}")
                            
                            if last_pattern_type == 'LH':
                                # Arm the LL-LH sequence for a potential Short
                                ll_lh_sequence_armed = True
                                armed_ll_price = lowest_low
                                logger.info(f"[Pattern] LL-LH sequence formed at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                          f"Price: {r.straddle_price:.2f}, VWAP: {current_vwap:.2f}")
                                lh_debug = f", LH: {lowest_high:.2f}" if lowest_high is not None else ""
                                logger.info(f"[Pattern] LL: {lowest_low:.2f}{lh_debug}")
                                logger.debug(f"[DEBUG] LL-LH sequence armed with LL={armed_ll_price}")
                                # Reset bullish sequence when bearish structure appears
                                hh_hl_sequence_formed = False
                                
                                # Track previous LL before updating
                                prev_ll = prev_lowest_low if prev_lowest_low is not None else lowest_low
                                
                                # Debug: Log relevant values before Short check
                                logger.debug(f"[DEBUG] Short check at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                           f"straddle_price={r.straddle_price}, prev_ll={prev_ll}, "
                                           f"ll_lh_sequence_formed={ll_lh_sequence_formed}")
                                
                                # Trigger Short immediately if conditions are met
                                if (ll_lh_sequence_formed and not in_short and not in_long1 and not in_long2 
                                    and r.straddle_price < current_vwap and r.straddle_price < prev_ll):
                                    logger.info(f"[Pattern] Straddle triggered Short at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
                                              f"Price: {r.straddle_price:.2f}, LL1: {lowest_low:.2f}, VWAP: {current_vwap:.2f}")
                                    df.loc[df['timestamp'] == r.timestamp, 'signal'] = 'Short'
                                    in_short = True
                                    short_entry_price = r.straddle_price
                                    logger.info(f"[SignalGenerator] Signal generated: Short at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} price {r.straddle_price:.2f}")
                                    
                                    # Log previous LL and LH values if they exist
                                    ll_debug = f"Previous LL: {lowest_low:.2f}" if lowest_low is not None else "No previous LL"
                                    lh_debug = f", Previous LH: {lowest_high:.2f}" if lowest_high is not None else ""
                                    logger.info(f"[Pattern] {ll_debug}{lh_debug}")
                                    
                                    # Reset sequence after signal
                                    ll_lh_sequence_formed = False
                                else:
                                    logger.debug(f"[Pattern] Short not triggered: r.straddle_price={r.straddle_price:.2f} is not below LL1={lowest_low:.2f}")
                            
                            last_pattern_type = 'LL'
                            has_ll_occurred = True  # Mark that a LL has occurred
                            
                    except Exception as e:
                        logger.error(f"[ERROR] Error in Lower Low detection: {str(e)}", exc_info=True)
                # New Lower High
                elif has_ll_occurred and lowest_low is not None and r.straddle_price > lowest_low and (lowest_high is None or r.straddle_price < lowest_high) and r.straddle_price < getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):
                    lowest_high = r.straddle_price
                    logger.info(f"[Pattern] New Lower High (LH) at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, VWAP: {getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                    if last_pattern_type == 'LL':
                        # Arm the LL-LH sequence for a potential Short
                        ll_lh_sequence_armed = True
                        armed_ll_price = lowest_low
                        logger.info(f"[Pattern] LL-LH sequence formed at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, VWAP: {getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                        logger.info(f"[Pattern] LL: {lowest_low:.2f}, LH: {lowest_high:.2f}")
                        logger.debug(f"[DEBUG] LL-LH sequence armed with LL={armed_ll_price}")
                    last_pattern_type = 'LH'
                else:
                    # Reset sequence if an unrelated pattern appears
                    last_pattern_type = None
                
                # Long1 entry (fixed SL/TP)
                if not in_long1 and hh_hl_sequence_formed:
                    # Only allow Long1 if price crosses above HH1 and is above VWAP
                    if r.straddle_price > highest_high and r.straddle_price > getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):
                        logger.info(f"[Pattern] Straddle crossed above HH1 at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, HH1: {highest_high:.2f}, VWAP: {getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                        df.loc[df['timestamp'] == r.timestamp, 'signal'] = 'Long1'
                        in_long1 = True
                        long1_entry_price = r.straddle_price
                        logger.info(f"[SignalGenerator] Signal generated: Long1 (fixed SL/TP) at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} price {r.straddle_price}")
                        logger.info(f"[SignalGenerator] Signal generated: Long2 (trail TP) at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} price {r.straddle_price}")
                        logger.info(f"[Pattern] Previous HH: {highest_high:.2f}, Previous HL: {highest_low:.2f}")
                        # Set both Long1 and Long2 signals and states
                        df.loc[df['timestamp'] == r.timestamp, 'signal'] = 'Long1,Long2'
                        in_long2 = True
                        long2_entry_price = r.straddle_price
                        # Set up trail TP for Long2
                        trail_tp = r.straddle_price + self.tp_long + self.trail_offset
                        hh_hl_sequence_formed = False  # Reset sequence after signal

                # Long1 exit (fixed SL/TP)
                if in_long1:
                    if (r.straddle_price <= long1_entry_price - self.sl_long or
                        r.straddle_price >= long1_entry_price + self.tp_long):
                        logger.info(f"[SignalGenerator] Long1 exited at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} entry {long1_entry_price} exit {r.straddle_price} (SL/TP hit), P&L: {r.straddle_price - long1_entry_price:.2f}")
                        in_long1 = False
                        long1_exit_price = r.straddle_price
                # Long2 entry (trail TP) - only after Long1 exit
                if not in_long2 and not in_long1 and long1_exit_price is not None:
                    if r.straddle_price > long1_exit_price and r.straddle_price > getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):
                        df.loc[df['timestamp'] == r.timestamp, 'signal'] = 'Long2'
                        in_long2 = True
                        long2_entry_price = r.straddle_price
                        logger.info(f"[SignalGenerator] Signal generated: Long2 (trail TP) at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} price {r.straddle_price}")
                        # Reset trail TP
                        trail_tp = r.straddle_price + self.tp_long + self.trail_offset
                # Long2 trailing TP exit
                if in_long2:
                    if r.timestamp.time() >= pd.to_datetime('15:20').time():
                        logger.info(f"[SignalGenerator] Long2 exited at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} entry {long2_entry_price} exit {r.straddle_price} (EOD), P&L: {r.straddle_price - long2_entry_price:.2f}")
                        in_long2 = False
                        long1_exit_price = None
                        long2_entry_price = None
                        trail_tp = None
                    # Update trailing TP if price moves in favor
                    if long2_entry_price is not None and trail_tp is not None and r.straddle_price > long2_entry_price and r.straddle_price + self.trail_offset > trail_tp:
                        trail_tp = r.straddle_price + self.trail_offset
                        logger.info(f"[SignalGenerator] Long2 trail TP updated to {trail_tp:.2f} at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    # Exit if price hits trailing TP or SL
                    if long2_entry_price is not None and r.straddle_price <= long2_entry_price - self.sl_long:
                        logger.info(f"[SignalGenerator] Long2 exited at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} entry {long2_entry_price} exit {r.straddle_price} (SL hit), P&L: {r.straddle_price - long2_entry_price:.2f}")
                        in_long2 = False
                        long1_exit_price = None
                        long2_entry_price = None
                        trail_tp = None
                    elif trail_tp is not None and r.straddle_price >= trail_tp:
                        logger.info(f"[SignalGenerator] Long2 exited at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} entry {long2_entry_price} exit {r.straddle_price} (Trail TP hit), P&L: {r.straddle_price - long2_entry_price:.2f}")
                        in_long2 = False
                        long1_exit_price = None
                        long2_entry_price = None
                        trail_tp = None
                
                # Short signal conditions        
                elif not in_short and not in_long1 and not in_long2 and r.straddle_price < getattr(r, 'vwap', getattr(r, 'vwap_5min', None)) and r.timestamp.time() < pd.to_datetime('15:20').time():
                    if (ll_lh_sequence_formed and 
                        (r.straddle_price < lowest_low or r.straddle_price < p2.straddle_price)):
                        # Log explicit crossing below LL1
                        if r.straddle_price < lowest_low:
                            logger.info(f"[Pattern] Straddle crossed below LL1 at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: Price: {r.straddle_price:.2f}, LL1: {lowest_low:.2f}, VWAP: {getattr(r, 'vwap', getattr(r, 'vwap_5min', None)):.2f}")
                        df.loc[df['timestamp'] == r.timestamp, 'signal'] = 'Short'
                        in_short = True
                        short_entry_price = r.straddle_price
                        lowest_low = r.straddle_price
                        logger.info(f"[SignalGenerator] Signal generated: Short at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} price {r.straddle_price}")
                        logger.info(f"[Pattern] Previous LL: {lowest_low:.2f}, Previous LH: {lowest_high:.2f}")
                # Short SL/TP checks
                if in_short:
                    if (r.straddle_price >= short_entry_price + self.sl_short or
                        r.straddle_price <= short_entry_price - self.tp_short):
                        logger.info(f"[SignalGenerator] Short exited at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} entry {short_entry_price} exit {r.straddle_price} (SL/TP hit), P&L: {short_entry_price - r.straddle_price:.2f}")
                        in_short = False
                        logger.info(f"[SignalGenerator] Short exited at {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')} entry {short_entry_price} exit {r.straddle_price} (EOD), P&L: {short_entry_price - r.straddle_price:.2f}")
                        short_entry_price = None
        return df