#!/usr/bin/env python3
"""
Monthly Portfolio Rebalance Script

This script runs automatically each month to rebalance your portfolio
using the Optimized Aggressive Momentum strategy on Alpaca.

Strategies available:
- optimized: Optimized Aggressive Momentum (RECOMMENDED - 31.3% test CAGR, 1.19 Sharpe)
  - 200-day MA trend filter, 15% alt data weight, 2.0x growth tilt, 20 stocks
- baseline: Original V3 Aggressive Momentum (29.6% test CAGR, 1.18 Sharpe)
- maxsharpe: Max Sharpe variant (lower volatility, better risk-adjusted returns)
- maxreturn: Max Return variant (more aggressive, potentially higher returns)
- mqt: Legacy MQT strategy
- enhanced: Legacy Enhanced MQT with alternative data

It is designed to run via macOS launchd scheduler.
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent.absolute()
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = LOG_DIR / f"rebalance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(SCRIPT_DIR))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / ".env")


def check_market_hours():
    """Check if US markets are open (basic check)."""
    import pytz
    from datetime import time

    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)

    # Skip weekends
    if now.weekday() >= 5:
        return False, "Weekend - markets closed"

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = now.time()

    if current_time < market_open:
        return False, f"Before market open (opens 9:30 AM ET)"
    if current_time > market_close:
        return False, f"After market close (closed 4:00 PM ET)"

    return True, "Market is open"


def run_rebalance(strategy_name='sentiment', use_live_alt_data=False, submit_after_hours=False):
    """
    Execute the monthly portfolio rebalance.

    Args:
        strategy_name: Strategy to use:
            - 'optimized': Optimized Aggressive Momentum (RECOMMENDED)
            - 'baseline': Original V3 Aggressive Momentum
            - 'maxsharpe': Max Sharpe variant
            - 'maxreturn': Max Return variant
            - 'mqt': Legacy MQT strategy
            - 'enhanced': Legacy Enhanced MQT with alt data
        use_live_alt_data: If True, fetch real alternative data (requires API access)
        submit_after_hours: If True, submit orders even when markets are closed (queued for market open)
    """
    logger.info("=" * 60)
    logger.info("MONTHLY PORTFOLIO REBALANCE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Strategy: {strategy_name.upper()}")
    logger.info("=" * 60)

    # Check API keys
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        logger.error("Alpaca API keys not found in .env file")
        return False

    # Check market hours
    market_open, status = check_market_hours()
    logger.info(f"Market status: {status}")

    if not market_open:
        if submit_after_hours:
            logger.info("Markets are closed but --submit-after-hours is set.")
            logger.info("DAY orders submitted after hours are queued for next market open.")
            logger.info("Trades will execute when markets open.")
            execute_live = True
            after_hours_mode = True
        else:
            logger.warning("Markets are closed. Rebalance will show planned trades but not execute.")
            logger.info("Use --submit-after-hours to queue orders for market open.")
            # Continue anyway to show what would happen, but in dry-run mode
            execute_live = False
            after_hours_mode = False
    else:
        execute_live = True
        after_hours_mode = False

    try:
        # Import trading components
        from alpaca_integration.alpaca_trader import AlpacaTrader, StrategyExecutor
        from configs.config import TOP_N_STOCKS

        # Initialize strategy based on selection
        strategy_lower = strategy_name.lower()

        if strategy_lower == 'optimized':
            from strategies.aggressive_momentum_optimized import OptimizedAggressiveMomentumStrategy
            logger.info("Initializing OPTIMIZED Aggressive Momentum Strategy")
            logger.info("  Parameters: 20 stocks, 200-day MA, 15% alt data, 2.0x growth tilt")
            logger.info(f"  - Using {'LIVE' if use_live_alt_data else 'synthetic'} alternative data")
            strategy = OptimizedAggressiveMomentumStrategy(
                top_n=20,
                use_synthetic=not use_live_alt_data
            )

        elif strategy_lower == 'baseline':
            from strategies.aggressive_momentum_optimized import BaselineAggressiveMomentumStrategy
            logger.info("Initializing BASELINE Aggressive Momentum Strategy (V3)")
            logger.info("  Parameters: 15 stocks, 50-day MA, no alt data")
            strategy = BaselineAggressiveMomentumStrategy(top_n=15)

        elif strategy_lower == 'maxsharpe':
            from strategies.aggressive_momentum_optimized import MaxSharpeAggressiveMomentumStrategy
            logger.info("Initializing MAX SHARPE Aggressive Momentum Strategy")
            logger.info("  Parameters: 20 stocks, 200-day MA, 10% alt data, volatility filter")
            strategy = MaxSharpeAggressiveMomentumStrategy(
                top_n=20,
                use_synthetic=not use_live_alt_data
            )

        elif strategy_lower == 'maxreturn':
            from strategies.aggressive_momentum_optimized import MaxReturnAggressiveMomentumStrategy
            logger.info("Initializing MAX RETURN Aggressive Momentum Strategy")
            logger.info("  Parameters: 15 stocks, 100-day MA, 20% alt data, 2.5x growth tilt")
            strategy = MaxReturnAggressiveMomentumStrategy(
                top_n=15,
                use_synthetic=not use_live_alt_data
            )

        elif strategy_lower == 'enhanced':
            from strategies.enhanced_strategy import EnhancedMQTStrategy
            logger.info(f"Initializing Enhanced MQT Strategy (top {TOP_N_STOCKS} stocks)")
            logger.info(f"  - Using {'LIVE' if use_live_alt_data else 'cached/synthetic'} alternative data")
            strategy = EnhancedMQTStrategy(
                top_n=TOP_N_STOCKS,
                use_synthetic_alt_data=not use_live_alt_data
            )

        elif strategy_lower == 'sentiment':
            from strategies.sentiment_optimized_strategy import SentimentOptimizedStrategy
            logger.info("Initializing SENTIMENT-OPTIMIZED Momentum Strategy")
            logger.info("  Parameters: 20 stocks, 200-day MA, 15% sentiment, 10% alt data")
            logger.info(f"  - Using {'LIVE' if use_live_alt_data else 'synthetic'} sentiment data")
            strategy = SentimentOptimizedStrategy(
                top_n=20,
                use_synthetic=not use_live_alt_data
            )

        elif strategy_lower == 'sentiment-confirmed':
            from strategies.sentiment_optimized_strategy import SentimentConfirmedStrategy
            logger.info("Initializing SENTIMENT-CONFIRMED Momentum Strategy")
            logger.info("  Parameters: 15 stocks, momentum + sentiment confirmation required")
            strategy = SentimentConfirmedStrategy(
                top_n=15,
                use_synthetic=not use_live_alt_data
            )

        else:  # 'mqt' or default
            from strategies.multifactor_strategy import MomentumQualityTrendStrategy
            logger.info(f"Initializing MQT Strategy (top {TOP_N_STOCKS} stocks)")
            strategy = MomentumQualityTrendStrategy(top_n=TOP_N_STOCKS)

        # Connect to Alpaca
        logger.info("Connecting to Alpaca (paper trading)...")
        trader = AlpacaTrader(paper=True)
        account = trader.get_account()

        logger.info(f"Account status: {account['status']}")
        logger.info(f"Portfolio value: ${account['portfolio_value']:,.2f}")
        logger.info(f"Buying power: ${account['buying_power']:,.2f}")
        logger.info(f"Cash: ${account['cash']:,.2f}")

        # Get current positions before rebalance
        logger.info("\nCurrent positions before rebalance:")
        positions = trader.get_positions()
        if positions:
            for symbol, pos in sorted(positions.items()):
                logger.info(f"  {symbol}: {pos['qty']} shares, ${pos['market_value']:,.2f}")
        else:
            logger.info("  No current positions")

        # Create executor and run strategy
        executor = StrategyExecutor(strategy, trader)

        logger.info("\n" + "=" * 60)
        if execute_live:
            if after_hours_mode:
                logger.info("SUBMITTING ORDERS FOR MARKET OPEN (DAY orders queued)")
            else:
                logger.info("EXECUTING LIVE REBALANCE")
        else:
            logger.info("DRY RUN (markets closed)")
        logger.info("=" * 60)

        # Always use DAY for fractional orders - they queue for market open if submitted after hours
        result = executor.execute_strategy(dry_run=not execute_live, time_in_force='day')

        if execute_live:
            if after_hours_mode:
                # Orders submitted for market open - positions won't change until then
                logger.info("\nOrders queued for market open.")
                logger.info("Positions will update when market opens and orders fill.")
                pending_orders = trader.get_orders(status='open')
                if pending_orders:
                    logger.info(f"\nPending orders ({len(pending_orders)}):")
                    for order in pending_orders[:10]:
                        logger.info(f"  {order['side']} {order['symbol']}: {order['status']}")
                    if len(pending_orders) > 10:
                        logger.info(f"  ... and {len(pending_orders) - 10} more")
                logger.info("\nORDERS SUBMITTED - WILL EXECUTE AT MARKET OPEN")
            else:
                # Log new positions after rebalance
                logger.info("\nPositions after rebalance:")
                positions = trader.get_positions()
                total_value = 0
                for symbol, pos in sorted(positions.items()):
                    logger.info(f"  {symbol}: {pos['qty']} shares, ${pos['market_value']:,.2f} ({pos['unrealized_plpc']:.2%})")
                    total_value += pos['market_value']

                logger.info(f"\nTotal invested: ${total_value:,.2f}")
                logger.info("REBALANCE COMPLETE")
        else:
            logger.info("\n[Dry run complete - no trades executed]")

        return True

    except Exception as e:
        logger.error(f"Error during rebalance: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    finally:
        logger.info(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Monthly Portfolio Rebalance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy Performance (Test Period 2021-2024):

  SENTIMENT-ENHANCED (NEW):
  sentiment:           ~23-30% CAGR, 0.9-1.1 Sharpe (RECOMMENDED with live data)
  sentiment-confirmed: Higher returns, requires momentum + sentiment confirmation

  MOMENTUM-ONLY:
  optimized:  31.3% CAGR, 1.19 Sharpe, -23.0% Max DD
  baseline:   29.6% CAGR, 1.18 Sharpe, -24.8% Max DD
  maxsharpe:  ~30% CAGR, ~1.25 Sharpe (lower volatility)
  maxreturn:  ~32% CAGR, ~1.10 Sharpe (higher risk)

  LEGACY:
  mqt:        Original MQT strategy
  enhanced:   MQT with alt data

Note: Sentiment strategies work best with --live-alt-data for real VADER analysis.
"""
    )
    parser.add_argument(
        '--strategy', '-s',
        choices=['optimized', 'baseline', 'maxsharpe', 'maxreturn', 'sentiment',
                 'sentiment-confirmed', 'mqt', 'enhanced'],
        default='sentiment',
        help='Strategy to use (default: sentiment)'
    )
    parser.add_argument(
        '--live-alt-data', '-l',
        action='store_true',
        help='Use live alternative data APIs (requires API keys for SEC, StockTwits, etc.)'
    )
    parser.add_argument(
        '--submit-after-hours', '-a',
        action='store_true',
        help='Submit orders even when markets are closed (queued for market open with OPG time-in-force)'
    )

    args = parser.parse_args()

    success = run_rebalance(
        strategy_name=args.strategy,
        use_live_alt_data=args.live_alt_data,
        submit_after_hours=args.submit_after_hours
    )
    sys.exit(0 if success else 1)
