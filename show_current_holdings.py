#!/usr/bin/env python3
"""
Show Current Stock Selections for Sentiment-Optimized Strategy

Displays the stocks that would be selected today using the sentiment strategy.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 70)
    print("SENTIMENT-OPTIMIZED STRATEGY - CURRENT STOCK SELECTIONS")
    print("=" * 70)

    # Load V3 data
    prices = pd.read_parquet('data/v3_full_prices.parquet')
    sector_map_df = pd.read_parquet('data/v3_sector_map.parquet')
    sector_map = dict(zip(sector_map_df['ticker'], sector_map_df['sector']))

    latest_date = prices.index[-1]
    print(f"\nData as of: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Universe: {len(prices.columns)} stocks")

    # Generate caches
    print("\nGenerating alternative data and sentiment caches...")
    from data.alternative_data import generate_synthetic_alternative_data
    from data.sentiment_analyzer import get_sentiment_scores_for_backtest

    tickers = prices.columns.tolist()
    alt_cache = generate_synthetic_alternative_data(tickers, prices, seed=42)
    sentiment_cache = get_sentiment_scores_for_backtest(tickers, prices, use_synthetic=True, seed=42)

    # Import strategies
    from strategies.sentiment_optimized_strategy import (
        strategy_sentiment_optimized,
        strategy_sentiment_confirmed
    )
    from strategies.aggressive_momentum_optimized import (
        strategy_aggressive_momentum_optimized
    )

    # Get holdings for each strategy
    print("\n" + "=" * 70)
    print("1. SENTIMENT-OPTIMIZED STRATEGY (NEW DEFAULT)")
    print("   75% Momentum + 15% Sentiment + 10% Alt Data")
    print("=" * 70)

    sentiment_holdings = strategy_sentiment_optimized(
        latest_date, prices, top_n=20, sector_map=sector_map,
        sentiment_cache=sentiment_cache, alt_data_cache=alt_cache,
        sentiment_weight=0.15, alt_weight=0.10, growth_tilt=2.0, ma_length=200
    )

    print(f"\nSelected {len(sentiment_holdings)} stocks:")
    print("-" * 50)
    print(f"{'Ticker':<8} {'Weight':<10} {'Sector':<30}")
    print("-" * 50)

    sentiment_sectors = {}
    for ticker, weight in sorted(sentiment_holdings.items(), key=lambda x: -x[1]):
        sector = sector_map.get(ticker, 'Unknown')
        print(f"{ticker:<8} {weight:>8.1%}   {sector}")
        sentiment_sectors[sector] = sentiment_sectors.get(sector, 0) + weight

    print("\nSector Allocation:")
    for sector, weight in sorted(sentiment_sectors.items(), key=lambda x: -x[1]):
        print(f"  {sector}: {weight:.1%}")

    # Sentiment-Confirmed (Aggressive)
    print("\n" + "=" * 70)
    print("2. SENTIMENT-CONFIRMED STRATEGY (AGGRESSIVE VARIANT)")
    print("   70% Momentum + 30% Sentiment, requires both signals strong")
    print("=" * 70)

    confirmed_holdings = strategy_sentiment_confirmed(
        latest_date, prices, top_n=15, sector_map=sector_map,
        sentiment_cache=sentiment_cache, alt_data_cache=alt_cache
    )

    print(f"\nSelected {len(confirmed_holdings)} stocks:")
    print("-" * 50)
    print(f"{'Ticker':<8} {'Weight':<10} {'Sector':<30}")
    print("-" * 50)

    for ticker, weight in sorted(confirmed_holdings.items(), key=lambda x: -x[1]):
        sector = sector_map.get(ticker, 'Unknown')
        print(f"{ticker:<8} {weight:>8.1%}   {sector}")

    # Previous Optimized (for comparison)
    print("\n" + "=" * 70)
    print("3. PREVIOUS OPTIMIZED STRATEGY (for comparison)")
    print("   85% Momentum + 15% Alt Data (no sentiment)")
    print("=" * 70)

    optimized_holdings = strategy_aggressive_momentum_optimized(
        latest_date, prices, top_n=20, sector_map=sector_map,
        alt_data_cache=alt_cache, alt_weight=0.15, growth_tilt=2.0, ma_length=200
    )

    print(f"\nSelected {len(optimized_holdings)} stocks:")
    print("-" * 50)
    print(f"{'Ticker':<8} {'Weight':<10} {'Sector':<30}")
    print("-" * 50)

    for ticker, weight in sorted(optimized_holdings.items(), key=lambda x: -x[1]):
        sector = sector_map.get(ticker, 'Unknown')
        print(f"{ticker:<8} {weight:>8.1%}   {sector}")

    # Overlap analysis
    print("\n" + "=" * 70)
    print("OVERLAP ANALYSIS")
    print("=" * 70)

    sentiment_set = set(sentiment_holdings.keys())
    optimized_set = set(optimized_holdings.keys())
    confirmed_set = set(confirmed_holdings.keys())

    print(f"\nSentiment vs Optimized:")
    print(f"  Common stocks: {len(sentiment_set & optimized_set)}")
    print(f"  Only in Sentiment: {sentiment_set - optimized_set}")
    print(f"  Only in Optimized: {optimized_set - sentiment_set}")

    print(f"\nSentiment-Confirmed vs Sentiment:")
    print(f"  Common stocks: {len(confirmed_set & sentiment_set)}")
    print(f"  Only in Confirmed: {confirmed_set - sentiment_set}")

    return sentiment_holdings


if __name__ == "__main__":
    holdings = main()
