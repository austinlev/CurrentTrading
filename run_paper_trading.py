"""
Paper Trading Execution Script

This script runs the selected optimal strategy on Alpaca paper trading.

Usage:
1. Set up Alpaca API keys:
   export ALPACA_API_KEY='your-key'
   export ALPACA_SECRET_KEY='your-secret'

2. Run:
   python run_paper_trading.py [--live]

The script will:
- Load the optimal strategy (MQT Strategy by default)
- Generate current signals
- Show the proposed trades
- Execute trades (if not in dry-run mode)
"""
import argparse
import os
import sys
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.multifactor_strategy import MomentumQualityTrendStrategy
from configs.config import TOP_N_STOCKS

# Check for Alpaca
try:
    from alpaca_integration.alpaca_trader import AlpacaTrader, StrategyExecutor
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Run paper trading strategy')
    parser.add_argument('--live', action='store_true',
                        help='Execute trades (default: dry run)')
    parser.add_argument('--strategy', type=str, default='MQT',
                        choices=['MQT', 'QualityMomentum', 'EnhancedMomentum'],
                        help='Strategy to use')
    parser.add_argument('--top-n', type=int, default=TOP_N_STOCKS,
                        help='Number of stocks to hold')

    args = parser.parse_args()

    print("=" * 60)
    print("ALPACA PAPER TRADING")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'LIVE TRADING' if args.live else 'DRY RUN'}")
    print("=" * 60)

    if not ALPACA_AVAILABLE:
        print("\nError: Alpaca libraries not installed.")
        print("Run: pip install alpaca-py")
        return

    # Check API keys
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("\nError: Alpaca API keys not found.")
        print("Please set environment variables:")
        print("  export ALPACA_API_KEY='your-key'")
        print("  export ALPACA_SECRET_KEY='your-secret'")
        return

    # Initialize strategy
    print(f"\nInitializing strategy: {args.strategy}")

    if args.strategy == 'MQT':
        from strategies.multifactor_strategy import MomentumQualityTrendStrategy
        strategy = MomentumQualityTrendStrategy(top_n=args.top_n)
    elif args.strategy == 'QualityMomentum':
        from strategies.multifactor_strategy import QualityMomentumStrategy
        strategy = QualityMomentumStrategy(top_n=args.top_n)
    else:
        from strategies.momentum_strategy import EnhancedMomentumStrategy
        strategy = EnhancedMomentumStrategy(top_n=args.top_n)

    # Initialize trader
    print("\nConnecting to Alpaca...")
    try:
        trader = AlpacaTrader(paper=True)
        account = trader.get_account()
        print(f"Connected! Account status: {account['status']}")
        print(f"Portfolio value: ${account['portfolio_value']:,.2f}")
    except Exception as e:
        print(f"Error connecting to Alpaca: {e}")
        return

    # Create executor
    executor = StrategyExecutor(strategy, trader)

    # Execute (dry run or live)
    print("\n" + "=" * 60)

    try:
        result = executor.execute_strategy(dry_run=not args.live)

        if args.live:
            print("\n" + "=" * 60)
            print("TRADES EXECUTED")
            print("=" * 60)

            # Show updated positions
            print("\nUpdated Positions:")
            positions = trader.get_positions()
            for symbol, pos in sorted(positions.items()):
                print(f"  {symbol}: ${pos['market_value']:,.2f} "
                      f"({pos['unrealized_plpc']:.2%} P/L)")
        else:
            print("\n[Dry run complete - no trades executed]")
            print("Run with --live flag to execute trades")

    except Exception as e:
        print(f"\nError executing strategy: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
