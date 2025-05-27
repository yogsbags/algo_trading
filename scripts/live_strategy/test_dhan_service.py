import asyncio
from dhan_quote_service import DhanQuoteService

async def test_dhan_service():
    # Test both NIFTY and BANKNIFTY
    for idx, underlying in enumerate(["NIFTY", "BANKNIFTY"]):
        service = DhanQuoteService(underlying=underlying)
        
        try:
            # Get the option chain ONCE
            chain = await service.get_option_chain()
            current_price = chain.get('last_price', 0)
            print(f"\n{'='*80}")
            print(f"{underlying} Last Price: {current_price:.2f}")
            print(f"{'='*80}")

            # Extract all strikes from the option chain data
            oc = chain.get('oc', {})
            if not oc:
                print("No option chain data available.")
                continue

            all_strikes = sorted(float(k) for k in oc.keys())
            if not all_strikes:
                print("No strikes found in option chain data.")
                continue

            # Find the ATM strike (closest to current_price)
            atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))
            atm_index = all_strikes.index(atm_strike)
            num_strikes = 5
            start = max(0, atm_index - num_strikes // 2)
            end = min(len(all_strikes), start + num_strikes)
            if end == len(all_strikes):
                start = max(0, end - num_strikes)
            selected_strikes = all_strikes[start:end]

            print(f"\n{num_strikes} ATM Strikes with Volume Data:")
            print("-" * 80)
            print(f"{'Strike':<10} | {'CE Price':<10} | {'CE Volume':<10} | {'PE Price':<10} | {'PE Volume':<10}")
            print("-" * 80)
            for strike in selected_strikes:
                strike_key = f"{strike:.6f}"
                data = oc.get(strike_key, {})
                ce = data.get('ce', {})
                pe = data.get('pe', {})
                ce_price = f"{ce.get('last_price', 0):.2f}" if ce else "N/A"
                ce_vol = ce.get('volume', 0) if ce else 0
                pe_price = f"{pe.get('last_price', 0):.2f}" if pe else "N/A"
                pe_vol = pe.get('volume', 0) if pe else 0
                print(f"{strike:<10.2f} | "
                      f"{ce_price:>9} | {ce_vol:>9,} | "
                      f"{pe_price:>9} | {pe_vol:>9,}")
        
        except Exception as e:
            print(f"Error testing {underlying}: {e}")
        finally:
            await service.close()
            print("\n")
        # Add delay between requests to avoid hitting rate limits
        if idx == 0:
            print('Waiting 5 seconds before next request to avoid rate limiting...')
            await asyncio.sleep(5)

# Run the test
if __name__ == "__main__":
    print("Testing DHAN Option Chain Service...")
    asyncio.run(test_dhan_service())
