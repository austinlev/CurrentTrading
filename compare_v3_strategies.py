"""
V3 Strategy Comparison Tool

Compares the V3 Ultimate Aggressive baseline against enhanced versions
with alternative data (insider trading, social sentiment, news).

Usage:
    python compare_v3_strategies.py
"""
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration from V3
TRAIN_START = "2010-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2024-12-31"
INITIAL_CAPITAL = 100000
TOP_N = 15

# Growth sectors for overweighting
GROWTH_SECTORS = ['Information Technology', 'Communication Services', 'Consumer Discretionary']

from strategies.v3_enhanced_strategy import V3_STRATEGIES
from data.alternative_data import generate_synthetic_alternative_data


def get_sp500_with_sectors():
    """Get S&P 500 tickers with sectors (excluding healthcare)"""
    from io import StringIO
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    df = tables[0][['Symbol', 'Security', 'GICS Sector']]
    df = df[~df['GICS Sector'].isin(['Health Care'])]
    return df


def download_data(tickers, start, end):
    """Download price data from yfinance"""
    clean_tickers = [t.replace('.', '-') for t in tickers]
    data = yf.download(clean_tickers, start=start, end=end, auto_adjust=True, progress=True)
    prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    prices = prices.ffill().bfill()
    missing = prices.isnull().sum() / len(prices)
    prices = prices[missing[missing < 0.1].index]
    return prices


class V3Backtester:
    """
    Backtester matching V3 workbook methodology:
    - Growth sector tilt
    - Optional regime filter
    - Lower transaction costs (aggressive assumption)
    """

    def __init__(self, prices, spy, sector_map, initial_capital=100000):
        self.prices = prices
        self.spy = spy
        self.sector_map = sector_map
        self.initial_capital = initial_capital
        self.commission = 0.0005
        self.slippage = 0.0005

    def apply_growth_tilt(self, weights, growth_bonus=1.3):
        """Overweight growth sectors"""
        if not weights:
            return weights

        adjusted = {}
        for ticker, weight in weights.items():
            sector = self.sector_map.get(ticker, 'Unknown')
            if sector in GROWTH_SECTORS:
                adjusted[ticker] = weight * growth_bonus
            else:
                adjusted[ticker] = weight

        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        return adjusted

    def run(self, signal_generator, start_date, end_date,
            rebalance_freq='ME', use_growth_tilt=True, alt_data_cache=None):
        """Run backtest"""
        mask = (self.prices.index >= start_date) & (self.prices.index <= end_date)
        prices = self.prices[mask].copy()

        rebalance_dates = prices.resample(rebalance_freq).last().index

        portfolio_values = []
        cash = self.initial_capital
        positions = {}

        for date in prices.index:
            stock_value = 0
            for ticker, pos in positions.items():
                if ticker in prices.columns:
                    price = prices.loc[date, ticker]
                    if not np.isnan(price):
                        stock_value += pos['shares'] * price

            portfolio_value = cash + stock_value
            portfolio_values.append(portfolio_value)

            if date in rebalance_dates:
                hist_mask = self.prices.index <= date
                hist_prices = self.prices[hist_mask]

                # Call strategy with alt_data_cache if provided
                if alt_data_cache is not None:
                    target_weights = signal_generator(
                        date, hist_prices,
                        sector_map=self.sector_map,
                        top_n=TOP_N,
                        alt_data_cache=alt_data_cache,
                        use_synthetic=True
                    )
                else:
                    target_weights = signal_generator(
                        date, hist_prices,
                        sector_map=self.sector_map,
                        top_n=TOP_N
                    )

                if use_growth_tilt:
                    target_weights = self.apply_growth_tilt(target_weights)

                total_weight = sum(target_weights.values()) if target_weights else 0
                if total_weight > 1.0:
                    target_weights = {k: v/total_weight for k, v in target_weights.items()}

                if target_weights:
                    # Sell all
                    for ticker in list(positions.keys()):
                        if ticker in prices.columns:
                            price = prices.loc[date, ticker]
                            if not np.isnan(price):
                                sell_price = price * (1 - self.slippage)
                                proceeds = positions[ticker]['shares'] * sell_price * (1 - self.commission)
                                cash += proceeds
                    positions = {}

                    # Buy new
                    for ticker, weight in target_weights.items():
                        if ticker in prices.columns and weight > 0:
                            price = prices.loc[date, ticker]
                            if not np.isnan(price) and price > 0:
                                buy_price = price * (1 + self.slippage)
                                target_value = portfolio_value * weight
                                shares = (target_value * (1 - self.commission)) / buy_price
                                cost = shares * buy_price * (1 + self.commission)
                                if cost <= cash and shares > 0:
                                    positions[ticker] = {
                                        'shares': shares,
                                        'entry_price': buy_price
                                    }
                                    cash -= cost

        return BacktestResults(prices.index.tolist(), portfolio_values, self.initial_capital)


class BacktestResults:
    """Results container matching V3 workbook"""

    def __init__(self, dates, values, initial_capital):
        self.portfolio_values = pd.Series(values, index=dates)
        self.initial_capital = initial_capital
        self.returns = self.portfolio_values.pct_change().dropna()

    def total_return(self):
        return (self.portfolio_values.iloc[-1] / self.initial_capital) - 1

    def cagr(self):
        years = len(self.portfolio_values) / 252
        return (self.portfolio_values.iloc[-1] / self.initial_capital) ** (1/years) - 1

    def volatility(self):
        return self.returns.std() * np.sqrt(252)

    def sharpe(self, rf=0.03):
        excess = self.returns - rf/252
        return excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    def sortino(self, rf=0.03):
        excess = self.returns - rf/252
        downside = excess[excess < 0]
        return excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    def max_drawdown(self):
        rolling_max = self.portfolio_values.expanding().max()
        return (self.portfolio_values / rolling_max - 1).min()

    def calmar(self):
        mdd = abs(self.max_drawdown())
        return self.cagr() / mdd if mdd > 0 else 0

    def summary(self):
        return {
            'CAGR': self.cagr(),
            'Volatility': self.volatility(),
            'Sharpe': self.sharpe(),
            'Sortino': self.sortino(),
            'Max Drawdown': self.max_drawdown(),
            'Calmar': self.calmar()
        }


def get_benchmark_stats(benchmark, start, end):
    """Calculate benchmark statistics"""
    bench = benchmark[(benchmark.index >= start) & (benchmark.index <= end)]
    if isinstance(bench, pd.DataFrame):
        bench = bench.iloc[:, 0]
    total_ret = (bench.iloc[-1] / bench.iloc[0]) - 1
    years = len(bench) / 252
    cagr = (1 + total_ret) ** (1/years) - 1
    returns = bench.pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.03) / vol if vol > 0 else 0
    rolling_max = bench.expanding().max()
    max_dd = (bench / rolling_max - 1).min()
    return {'CAGR': cagr, 'Volatility': vol, 'Sharpe': sharpe, 'Max Drawdown': max_dd, 'Calmar': cagr/abs(max_dd)}


def main():
    print("=" * 70)
    print("  V3 STRATEGY COMPARISON")
    print("  Baseline vs Enhanced Strategies with Alternative Data")
    print("=" * 70)

    # Load data
    print("\nLoading stock data...")
    tickers_df = get_sp500_with_sectors()
    print(f"  Universe: {len(tickers_df)} stocks (excluding Healthcare)")

    sector_map = dict(zip(
        tickers_df['Symbol'].str.replace('.', '-'),
        tickers_df['GICS Sector']
    ))

    all_tickers = tickers_df['Symbol'].tolist()
    full_prices = download_data(all_tickers, "2009-01-01", TEST_END)
    print(f"  Loaded {len(full_prices.columns)} stocks with sufficient data")

    # Download benchmarks
    print("\nDownloading benchmarks...")
    spy = yf.download('SPY', start="2009-01-01", end=TEST_END, auto_adjust=True, progress=False)['Close']
    qqq = yf.download('QQQ', start="2009-01-01", end=TEST_END, auto_adjust=True, progress=False)['Close']

    # Generate synthetic alternative data for backtesting
    print("\nGenerating synthetic alternative data for backtesting...")
    alt_data_cache = generate_synthetic_alternative_data(
        full_prices.columns.tolist(),
        full_prices,
        seed=42
    )

    # Initialize backtester
    bt = V3Backtester(full_prices, spy, sector_map, INITIAL_CAPITAL)

    # ========================================
    # TRAINING PERIOD
    # ========================================
    print("\n" + "=" * 70)
    print(f"  TRAINING PERIOD ({TRAIN_START} to {TRAIN_END})")
    print("  Settings: Monthly rebalance, Growth tilt ON")
    print("=" * 70)

    train_results = {}
    for name, strategy_func in V3_STRATEGIES.items():
        print(f"\n  Running {name}...")

        # Check if strategy needs alt_data_cache
        needs_alt_data = name != 'V3 Ultimate Aggressive'

        result = bt.run(
            strategy_func,
            TRAIN_START, TRAIN_END,
            rebalance_freq='ME',
            use_growth_tilt=True,
            alt_data_cache=alt_data_cache if needs_alt_data else None
        )
        train_results[name] = result
        s = result.summary()
        print(f"    CAGR: {s['CAGR']:.1%} | Sharpe: {s['Sharpe']:.2f} | MaxDD: {s['Max Drawdown']:.1%}")

    # ========================================
    # TEST PERIOD
    # ========================================
    print("\n" + "=" * 70)
    print(f"  TEST PERIOD ({TEST_START} to {TEST_END}) - OUT OF SAMPLE")
    print("=" * 70)

    test_results = {}
    for name, strategy_func in V3_STRATEGIES.items():
        print(f"\n  Running {name}...")

        needs_alt_data = name != 'V3 Ultimate Aggressive'

        result = bt.run(
            strategy_func,
            TEST_START, TEST_END,
            rebalance_freq='ME',
            use_growth_tilt=True,
            alt_data_cache=alt_data_cache if needs_alt_data else None
        )
        test_results[name] = result
        s = result.summary()
        print(f"    CAGR: {s['CAGR']:.1%} | Sharpe: {s['Sharpe']:.2f} | MaxDD: {s['Max Drawdown']:.1%}")

    # ========================================
    # RESULTS TABLES
    # ========================================
    print("\n" + "=" * 90)
    print("  TRAINING PERIOD RESULTS")
    print("=" * 90)

    spy_train = get_benchmark_stats(spy, TRAIN_START, TRAIN_END)
    qqq_train = get_benchmark_stats(qqq, TRAIN_START, TRAIN_END)

    rows = [
        {'Strategy': 'SPY (S&P 500)', **spy_train},
        {'Strategy': 'QQQ (Nasdaq)', **qqq_train}
    ]
    for name, result in train_results.items():
        rows.append({'Strategy': name, **result.summary()})

    train_df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    display_df = train_df.copy()
    for col in ['CAGR', 'Volatility', 'Max Drawdown']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
    for col in ['Sharpe', 'Calmar', 'Sortino']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    print(display_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("  TEST PERIOD RESULTS (OUT OF SAMPLE)")
    print("=" * 90)

    spy_test = get_benchmark_stats(spy, TEST_START, TEST_END)
    qqq_test = get_benchmark_stats(qqq, TEST_START, TEST_END)

    rows = [
        {'Strategy': 'SPY (S&P 500)', **spy_test},
        {'Strategy': 'QQQ (Nasdaq)', **qqq_test}
    ]
    for name, result in test_results.items():
        rows.append({'Strategy': name, **result.summary()})

    test_df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    display_df = test_df.copy()
    for col in ['CAGR', 'Volatility', 'Max Drawdown']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
    for col in ['Sharpe', 'Calmar', 'Sortino']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    print(display_df.to_string(index=False))

    # ========================================
    # IMPROVEMENT ANALYSIS
    # ========================================
    print("\n" + "=" * 70)
    print("  IMPROVEMENT vs V3 BASELINE")
    print("=" * 70)

    baseline_name = 'V3 Ultimate Aggressive'

    for period, results, bench_spy in [('Training', train_results, spy_train),
                                        ('Test', test_results, spy_test)]:
        print(f"\n{period} Period:")
        if baseline_name in results:
            baseline = results[baseline_name].summary()
            print(f"  Baseline (V3): CAGR {baseline['CAGR']:.1%}, Sharpe {baseline['Sharpe']:.2f}")
            print(f"  SPY Benchmark: CAGR {bench_spy['CAGR']:.1%}, Sharpe {bench_spy['Sharpe']:.2f}")
            print()

            for name, result in results.items():
                if name != baseline_name:
                    s = result.summary()
                    cagr_diff = s['CAGR'] - baseline['CAGR']
                    sharpe_diff = s['Sharpe'] - baseline['Sharpe']
                    dd_diff = s['Max Drawdown'] - baseline['Max Drawdown']

                    improvement = "✓" if sharpe_diff > 0 else "✗"
                    print(f"  {improvement} {name}:")
                    print(f"      Sharpe: {s['Sharpe']:.2f} ({sharpe_diff:+.2f})")
                    print(f"      CAGR: {s['CAGR']:.1%} ({cagr_diff:+.1%})")
                    print(f"      MaxDD: {s['Max Drawdown']:.1%} ({dd_diff:+.1%})")

    # ========================================
    # FINAL RANKING
    # ========================================
    print("\n" + "=" * 70)
    print("  FINAL STRATEGY RANKING")
    print("  (30% Training + 70% Test Sharpe)")
    print("=" * 70)

    strat_only_train = train_df[~train_df['Strategy'].isin(['SPY (S&P 500)', 'QQQ (Nasdaq)'])].copy()
    strat_only_test = test_df[~test_df['Strategy'].isin(['SPY (S&P 500)', 'QQQ (Nasdaq)'])].copy()

    combined = strat_only_train[['Strategy', 'Sharpe']].merge(
        strat_only_test[['Strategy', 'Sharpe']],
        on='Strategy',
        suffixes=('_train', '_test')
    )
    combined['Score'] = 0.3 * combined['Sharpe_train'] + 0.7 * combined['Sharpe_test']
    combined = combined.sort_values('Score', ascending=False)

    print("\n" + combined.to_string(index=False))

    best = combined.iloc[0]['Strategy']
    print(f"\n{'*' * 70}")
    print(f"  BEST STRATEGY: {best}")
    print(f"  Combined Score: {combined.iloc[0]['Score']:.3f}")
    print(f"{'*' * 70}")

    # Save results
    os.makedirs('results', exist_ok=True)
    train_df.to_csv('results/v3_comparison_training.csv', index=False)
    test_df.to_csv('results/v3_comparison_test.csv', index=False)
    combined.to_csv('results/v3_ranking.csv', index=False)
    print("\n  Results saved to results/v3_*.csv")

    return best, train_results, test_results


if __name__ == "__main__":
    best, train_results, test_results = main()
