"""
Main Backtest Runner

Runs comprehensive backtests of all strategies on training and test datasets.
Compares performance and identifies the best strategy.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    TRAIN_START_DATE, TRAIN_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    INITIAL_CAPITAL, TOP_N_STOCKS,
    BENCHMARK_TICKER
)
from data.data_loader import prepare_universe, get_benchmark_data
from backtest.backtester import (
    Backtester, get_monthly_rebalance_dates
)
from strategies.momentum_strategy import (
    MomentumStrategy,
    EnhancedMomentumStrategy,
    DualMomentumStrategy,
    TrendFollowingMomentumStrategy,
    VolatilityAdjustedMomentumStrategy
)
from strategies.multifactor_strategy import (
    MultiFactorStrategy,
    QualityMomentumStrategy,
    AdaptiveMultiFactorStrategy,
    MomentumQualityTrendStrategy
)
from strategies.mean_reversion_strategy import (
    RSIMeanReversionStrategy,
    OversoldBounceStrategy,
    CombinedMeanReversionStrategy
)

# Try to import ML strategies
try:
    from strategies.ml_strategy import MLStrategy, EnsembleMLStrategy, HybridStrategy
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Note: ML strategies not available (missing sklearn/xgboost/lightgbm)")

# Try to import enhanced strategies with alternative data
try:
    from strategies.enhanced_strategy import (
        EnhancedMQTStrategy,
        InsiderMomentumStrategy,
        SocialMomentumStrategy,
        AdaptiveAltDataStrategy,
        HighConvictionAltDataStrategy
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Note: Enhanced strategies not available")

# Try to import optimized aggressive momentum strategies
try:
    from strategies.aggressive_momentum_optimized import OPTIMIZED_STRATEGIES
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    print("Note: Optimized aggressive momentum strategies not available")


def get_all_strategies(top_n=TOP_N_STOCKS):
    """Get all strategies to test"""
    strategies = {
        # Momentum-based
        'Momentum (12-1)': MomentumStrategy(top_n=top_n),
        'Enhanced Momentum': EnhancedMomentumStrategy(top_n=top_n),
        'Dual Momentum': DualMomentumStrategy(top_n=top_n),
        'Trend Following Momentum': TrendFollowingMomentumStrategy(top_n=top_n),
        'Volatility-Adjusted Momentum': VolatilityAdjustedMomentumStrategy(top_n=top_n),

        # Multi-factor
        'Multi-Factor': MultiFactorStrategy(top_n=top_n),
        'Quality Momentum': QualityMomentumStrategy(top_n=top_n),
        'Adaptive Multi-Factor': AdaptiveMultiFactorStrategy(top_n=top_n),
        'MQT Strategy': MomentumQualityTrendStrategy(top_n=top_n),

        # Mean Reversion
        'RSI Mean Reversion': RSIMeanReversionStrategy(top_n=top_n),
        'Oversold Bounce': OversoldBounceStrategy(top_n=top_n),
        'Combined Mean Reversion': CombinedMeanReversionStrategy(top_n=top_n),
    }

    # Add ML strategies if available
    if ML_AVAILABLE:
        strategies['ML Strategy'] = MLStrategy(top_n=top_n)
        strategies['Ensemble ML'] = EnsembleMLStrategy(top_n=top_n)
        strategies['Hybrid ML-Factor'] = HybridStrategy(top_n=top_n)

    # Add enhanced strategies with alternative data
    if ENHANCED_AVAILABLE:
        strategies['Enhanced MQT + Alt Data'] = EnhancedMQTStrategy(
            top_n=top_n, use_synthetic_alt_data=True
        )
        strategies['Insider Momentum'] = InsiderMomentumStrategy(
            top_n=top_n, use_synthetic=True
        )
        strategies['Social Momentum'] = SocialMomentumStrategy(
            top_n=top_n, use_synthetic=True
        )
        strategies['Adaptive Alt Data'] = AdaptiveAltDataStrategy(
            top_n=top_n, use_synthetic=True
        )
        strategies['High Conviction Alt Data'] = HighConvictionAltDataStrategy(
            top_n=20, max_weight=0.07, use_synthetic=True  # Concentrated portfolio
        )

    # Add optimized aggressive momentum strategies
    if OPTIMIZED_AVAILABLE:
        from strategies.aggressive_momentum_optimized import (
            OptimizedAggressiveMomentumStrategy,
            BaselineAggressiveMomentumStrategy,
            MaxSharpeAggressiveMomentumStrategy,
            MaxReturnAggressiveMomentumStrategy
        )
        strategies['Optimized Aggressive Momentum'] = OptimizedAggressiveMomentumStrategy(top_n=20)
        strategies['Baseline Aggressive Momentum'] = BaselineAggressiveMomentumStrategy(top_n=15)
        strategies['Max Sharpe Aggressive Momentum'] = MaxSharpeAggressiveMomentumStrategy(top_n=20)
        strategies['Max Return Aggressive Momentum'] = MaxReturnAggressiveMomentumStrategy(top_n=15)

    return strategies


def run_single_backtest(strategy, prices, benchmark, name="Strategy"):
    """Run backtest for a single strategy"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    # Create backtester
    backtester = Backtester(prices, benchmark['Close'])

    # Get monthly rebalance dates
    rebalance_dates = get_monthly_rebalance_dates(prices)

    # Run backtest
    results = backtester.run(strategy, rebalance_dates)

    # Print summary
    results.print_summary()

    return results


def run_all_backtests(prices, benchmark, strategies):
    """Run backtests for all strategies"""
    results = {}

    for name, strategy in strategies.items():
        try:
            result = run_single_backtest(strategy, prices, benchmark, name)
            results[name] = result
        except Exception as e:
            print(f"Error running {name}: {e}")
            continue

    return results


def compare_strategies(results_dict, benchmark):
    """Compare all strategy results"""
    comparison = []

    # Calculate benchmark metrics
    bench_returns = benchmark['Close'].pct_change().dropna()
    bench_total_return = (benchmark['Close'].iloc[-1] / benchmark['Close'].iloc[0]) - 1
    years = len(benchmark) / 252
    bench_cagr = (1 + bench_total_return) ** (1 / years) - 1
    bench_vol = bench_returns.std() * np.sqrt(252)
    bench_sharpe = (bench_cagr - 0.04) / bench_vol

    rolling_max = benchmark['Close'].expanding().max()
    bench_drawdown = (benchmark['Close'] / rolling_max - 1).min()

    comparison.append({
        'Strategy': 'SPY (Benchmark)',
        'Total Return': bench_total_return,
        'CAGR': bench_cagr,
        'Volatility': bench_vol,
        'Sharpe Ratio': bench_sharpe,
        'Max Drawdown': bench_drawdown,
        'Calmar Ratio': bench_cagr / abs(bench_drawdown) if bench_drawdown != 0 else 0
    })

    for name, results in results_dict.items():
        comparison.append({
            'Strategy': name,
            'Total Return': results.total_return(),
            'CAGR': results.cagr(),
            'Volatility': results.volatility(),
            'Sharpe Ratio': results.sharpe_ratio(),
            'Max Drawdown': results.max_drawdown(),
            'Calmar Ratio': results.calmar_ratio()
        })

    df = pd.DataFrame(comparison)

    # Sort by Sharpe ratio
    df = df.sort_values('Sharpe Ratio', ascending=False)

    return df


def print_comparison_table(comparison_df):
    """Print formatted comparison table"""
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON (Sorted by Sharpe Ratio)")
    print("=" * 100)

    # Format percentages
    formatted = comparison_df.copy()
    formatted['Total Return'] = formatted['Total Return'].apply(lambda x: f"{x:.1%}")
    formatted['CAGR'] = formatted['CAGR'].apply(lambda x: f"{x:.1%}")
    formatted['Volatility'] = formatted['Volatility'].apply(lambda x: f"{x:.1%}")
    formatted['Sharpe Ratio'] = formatted['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
    formatted['Max Drawdown'] = formatted['Max Drawdown'].apply(lambda x: f"{x:.1%}")
    formatted['Calmar Ratio'] = formatted['Calmar Ratio'].apply(lambda x: f"{x:.2f}")

    print(formatted.to_string(index=False))
    print("=" * 100)


def select_best_strategy(comparison_df, results_dict):
    """Select the best strategy based on multiple criteria"""
    # Exclude benchmark
    strategies_only = comparison_df[comparison_df['Strategy'] != 'SPY (Benchmark)'].copy()

    # Scoring system
    # 1. Sharpe Ratio (40%)
    # 2. Calmar Ratio (25%)
    # 3. CAGR (20%)
    # 4. Max Drawdown (15%)

    strategies_only['Sharpe_Score'] = strategies_only['Sharpe Ratio'].rank(pct=True)
    strategies_only['Calmar_Score'] = strategies_only['Calmar Ratio'].rank(pct=True)
    strategies_only['CAGR_Score'] = strategies_only['CAGR'].rank(pct=True)
    strategies_only['DD_Score'] = (-strategies_only['Max Drawdown']).rank(pct=True)

    strategies_only['Total_Score'] = (
        0.40 * strategies_only['Sharpe_Score'] +
        0.25 * strategies_only['Calmar_Score'] +
        0.20 * strategies_only['CAGR_Score'] +
        0.15 * strategies_only['DD_Score']
    )

    best_name = strategies_only.loc[strategies_only['Total_Score'].idxmax(), 'Strategy']

    print(f"\n{'='*60}")
    print(f"BEST STRATEGY: {best_name}")
    print(f"{'='*60}")

    return best_name


def main():
    """Main execution function"""
    print("=" * 80)
    print("US STOCK ALGO TRADING STRATEGY BACKTEST")
    print("Excluding Healthcare Stocks")
    print("=" * 80)

    # ========================================
    # PHASE 1: Training Period (2010-2020)
    # ========================================
    print("\n" + "#" * 80)
    print("PHASE 1: TRAINING PERIOD BACKTEST")
    print(f"Period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")
    print("#" * 80)

    # Load training data
    print("\nLoading training data...")
    train_prices = prepare_universe(TRAIN_START_DATE, TRAIN_END_DATE)
    train_benchmark = get_benchmark_data(TRAIN_START_DATE, TRAIN_END_DATE)

    print(f"Loaded {len(train_prices.columns)} stocks")
    print(f"Date range: {train_prices.index[0]} to {train_prices.index[-1]}")

    # Get strategies
    strategies = get_all_strategies()
    print(f"\nTesting {len(strategies)} strategies...")

    # Run training backtests
    train_results = run_all_backtests(train_prices, train_benchmark, strategies)

    # Compare training results
    train_comparison = compare_strategies(train_results, train_benchmark)
    print_comparison_table(train_comparison)

    # ========================================
    # PHASE 2: Testing Period (2021-2024)
    # ========================================
    print("\n" + "#" * 80)
    print("PHASE 2: OUT-OF-SAMPLE TEST PERIOD")
    print(f"Period: {TEST_START_DATE} to {TEST_END_DATE}")
    print("#" * 80)

    # Load test data
    print("\nLoading test data...")
    test_prices = prepare_universe(TEST_START_DATE, TEST_END_DATE)
    test_benchmark = get_benchmark_data(TEST_START_DATE, TEST_END_DATE)

    print(f"Loaded {len(test_prices.columns)} stocks")
    print(f"Date range: {test_prices.index[0]} to {test_prices.index[-1]}")

    # Get fresh strategies (reset any state)
    strategies = get_all_strategies()

    # Run test backtests
    test_results = run_all_backtests(test_prices, test_benchmark, strategies)

    # Compare test results
    test_comparison = compare_strategies(test_results, test_benchmark)
    print_comparison_table(test_comparison)

    # ========================================
    # PHASE 3: Final Selection
    # ========================================
    print("\n" + "#" * 80)
    print("PHASE 3: STRATEGY SELECTION")
    print("#" * 80)

    # Find strategies that performed well in BOTH periods
    print("\n--- Training Period Top 5 ---")
    print(train_comparison[['Strategy', 'CAGR', 'Sharpe Ratio', 'Max Drawdown']].head(6))

    print("\n--- Test Period Top 5 ---")
    print(test_comparison[['Strategy', 'CAGR', 'Sharpe Ratio', 'Max Drawdown']].head(6))

    # Combined scoring
    # Weight both training and test performance
    train_scores = train_comparison[train_comparison['Strategy'] != 'SPY (Benchmark)'].copy()
    test_scores = test_comparison[test_comparison['Strategy'] != 'SPY (Benchmark)'].copy()

    train_scores['train_sharpe_rank'] = train_scores['Sharpe Ratio'].rank(pct=True)
    test_scores['test_sharpe_rank'] = test_scores['Sharpe Ratio'].rank(pct=True)

    # Merge scores
    combined = train_scores[['Strategy', 'train_sharpe_rank']].merge(
        test_scores[['Strategy', 'test_sharpe_rank']],
        on='Strategy'
    )

    # Combined score: 40% train, 60% test (favor out-of-sample)
    combined['combined_score'] = 0.4 * combined['train_sharpe_rank'] + 0.6 * combined['test_sharpe_rank']
    combined = combined.sort_values('combined_score', ascending=False)

    print("\n--- Combined Ranking (40% train, 60% test) ---")
    print(combined)

    best_strategy_name = combined.iloc[0]['Strategy']

    print(f"\n{'*'*60}")
    print(f"RECOMMENDED STRATEGY: {best_strategy_name}")
    print(f"{'*'*60}")

    # Print detailed stats for best strategy
    print("\n--- Best Strategy Training Period Stats ---")
    if best_strategy_name in train_results:
        train_results[best_strategy_name].print_summary()

    print("\n--- Best Strategy Test Period Stats ---")
    if best_strategy_name in test_results:
        test_results[best_strategy_name].print_summary()

    # Save results
    os.makedirs('results', exist_ok=True)
    train_comparison.to_csv('results/training_comparison.csv', index=False)
    test_comparison.to_csv('results/test_comparison.csv', index=False)
    combined.to_csv('results/combined_ranking.csv', index=False)

    print("\nResults saved to 'results/' directory")

    return best_strategy_name, train_results, test_results


if __name__ == "__main__":
    best_strategy, train_results, test_results = main()
