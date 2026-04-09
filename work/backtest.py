"""
backtest.py — Portfolio Backtesting Engine (Fixed)
===================================================
Loads raw stock CSVs, computes actual weekly returns,
reads model predictions, simulates weekly portfolio allocation,
and outputs P&L vs benchmarks.

Place in: work/backtest.py
Run from: work/ folder  →  python backtest.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
INITIAL_CAPITAL = 10_000
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
OUT_DIR = '../data/results'
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    'AAPL': '../data/Apple.csv',
    'MSFT': '../data/MSFT.csv',
    'GOOGL': '../data/google.csv',
    'AMZN': '../data/Amazon.csv',
    'NVDA': '../data/NVIDIA.csv',
}

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA AND COMPUTE ACTUAL WEEKLY RETURNS
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("BACKTEST — Loading Raw Data")
print("=" * 60)

frames = []
for ticker, path in FILES.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'].astype(str).str.replace(',', ''), errors='coerce')
    df = df.sort_values('Date').dropna(subset=['Close']).set_index('Date')

    w = pd.DataFrame()
    w['Close'] = df['Close'].resample('W-FRI').last()
    w['Weekly_Return'] = w['Close'].pct_change()  # decimal: 0.02 = 2%
    w['Ticker'] = ticker
    w = w.dropna().reset_index()
    frames.append(w)

all_weekly = pd.concat(frames).sort_values(['Date', 'Ticker']).reset_index(drop=True)

# Filter to test period (2024+)
test_data = all_weekly[all_weekly['Date'] >= '2024-01-01'].copy()
print(f"✅ Loaded raw data — {len(test_data)} rows, "
      f"{test_data['Date'].nunique()} weeks")

# Pivot actual returns
actual_returns = test_data.pivot_table(index='Date', columns='Ticker',
                                        values='Weekly_Return').sort_index()
weeks = actual_returns.index
print(f"   Test period: {weeks[0].date()} → {weeks[-1].date()}")
print(f"   Avg weekly return: {actual_returns.mean().mean():.4f} ({actual_returns.mean().mean()*100:.2f}%)")


# ══════════════════════════════════════════════════════════════════════
# 2. RANKING → WEIGHT CONVERSION
# ══════════════════════════════════════════════════════════════════════
def ranks_to_weights(rank_df, method='linear'):
    weights = pd.DataFrame(index=rank_df.index, columns=rank_df.columns, dtype=float)
    if method == 'linear':
        inv = (len(TICKERS) + 1) - rank_df
        weights = inv.div(inv.sum(axis=1), axis=0)
    elif method == 'top2':
        for idx in rank_df.index:
            row = rank_df.loc[idx]
            top2 = row.nsmallest(2).index
            weights.loc[idx] = 0.0
            weights.loc[idx, top2] = 0.5
    elif method == 'winner':
        for idx in rank_df.index:
            row = rank_df.loc[idx]
            winner = row.idxmin()
            weights.loc[idx] = 0.0
            weights.loc[idx, winner] = 1.0
    return weights


# ══════════════════════════════════════════════════════════════════════
# 3. LOAD MODEL PREDICTIONS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BACKTEST — Loading Predictions")
print("=" * 60)

def load_predictions(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    pivot = df.pivot_table(index='Date', columns='Ticker', values='Predicted_Return')
    ranks = pivot.rank(axis=1, ascending=False)
    return ranks.sort_index()

models = {}

pred_files = {
    'ARIMA':   '../data/results/arima_predictions.csv',
    'XGBoost': '../data/results/xgboost_predictions.csv',
    'LSTM':    '../data/results/lstm_predictions.csv',
}

for name, path in pred_files.items():
    if os.path.exists(path):
        models[name] = load_predictions(path)
        print(f"✅ Loaded {name} predictions ({len(models[name])} weeks)")

if not models:
    print("\n⚠️  No prediction files found!")
    print("   Expected paths:")
    for name, path in pred_files.items():
        print(f"   {name}: {path}")
    print("\n   Using equal-weight only.")

# Benchmarks
equal_ranks = pd.DataFrame(1, index=weeks, columns=TICKERS)
models['Equal Weight'] = equal_ranks
models['Oracle (perfect)'] = actual_returns.rank(axis=1, ascending=False)


# ══════════════════════════════════════════════════════════════════════
# 4. SIMULATE PORTFOLIO
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BACKTEST — Simulating Portfolios")
print("=" * 60)

def simulate(weights_df, actual_ret, initial=INITIAL_CAPITAL):
    common = weights_df.index.intersection(actual_ret.index)
    w = weights_df.loc[common]
    r = actual_ret.loc[common]
    weekly_ret = (w * r).sum(axis=1)
    cum_ret = (1 + weekly_ret).cumprod()
    port_val = initial * cum_ret
    result = pd.DataFrame({
        'Week_Return': weekly_ret,
        'Cumulative_Return': cum_ret - 1,
        'Portfolio_Value': port_val
    }, index=common)
    return result

results = {}
for name, rank_df in models.items():
    common = rank_df.index.intersection(weeks)
    if len(common) == 0:
        print(f"⚠️  {name}: no overlapping dates, skipping")
        continue
    weights = ranks_to_weights(rank_df.loc[common], method='linear')
    results[name] = simulate(weights, actual_returns)
    final_val = results[name]['Portfolio_Value'].iloc[-1]
    total_ret = results[name]['Cumulative_Return'].iloc[-1] * 100
    print(f"   {name:25s} → Final: ${final_val:,.2f}  |  Return: {total_ret:+.1f}%")


# ══════════════════════════════════════════════════════════════════════
# 5. COMPUTE METRICS
# ══════════════════════════════════════════════════════════════════════
def compute_metrics(result_df):
    wr = result_df['Week_Return']
    total_return = result_df['Cumulative_Return'].iloc[-1] * 100
    n_weeks = len(wr)
    annual_return = ((1 + result_df['Cumulative_Return'].iloc[-1]) **
                     (52 / max(n_weeks, 1)) - 1) * 100
    sharpe = (wr.mean() / wr.std()) * np.sqrt(52) if wr.std() > 0 else 0
    cum = (1 + wr).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min() * 100
    win_rate = (wr > 0).mean() * 100
    return {
        'Total Return (%)': round(total_return, 2),
        'Annualized Return (%)': round(annual_return, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown (%)': round(max_dd, 2),
        'Win Rate (%)': round(win_rate, 1),
        'Final Value ($)': round(result_df['Portfolio_Value'].iloc[-1], 2),
        'Profit/Loss ($)': round(result_df['Portfolio_Value'].iloc[-1] - INITIAL_CAPITAL, 2),
        'Weeks': n_weeks,
    }

summary = {}
for name, res in results.items():
    summary[name] = compute_metrics(res)

summary_df = pd.DataFrame(summary).T
summary_df.index.name = 'Model'

print("\n" + "=" * 70)
print("BACKTEST SUMMARY")
print("=" * 70)
print(summary_df.to_string())


# ══════════════════════════════════════════════════════════════════════
# 6. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
weekly_all = pd.DataFrame(index=weeks)
for name, res in results.items():
    weekly_all[name] = res['Week_Return']
weekly_all.to_csv(f'{OUT_DIR}/backtest_results.csv')
summary_df.to_csv(f'{OUT_DIR}/backtest_summary.csv')
print(f"\n✅ Saved: {OUT_DIR}/backtest_results.csv")
print(f"✅ Saved: {OUT_DIR}/backtest_summary.csv")


# ══════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════════
# Cumulative portfolio value
fig, ax = plt.subplots(figsize=(12, 6))
for name, res in results.items():
    if 'Oracle' in name:
        style, lw = ':', 1
    elif 'Equal' in name:
        style, lw = '--', 1.5
    else:
        style, lw = '-', 2
    ax.plot(res.index, res['Portfolio_Value'], label=name, linestyle=style, linewidth=lw)

ax.set_title(f'Portfolio Value Over Test Period (${INITIAL_CAPITAL:,} initial)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($)')
ax.legend(loc='upper left')
ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/backtest_cumulative.png', dpi=150)
print(f"✅ Saved: {OUT_DIR}/backtest_cumulative.png")

# Metrics bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
for ax, metric in zip(axes, metrics_to_plot):
    vals = summary_df[metric]
    colors = ['green' if v > 0 else 'red' for v in vals]
    if metric == 'Max Drawdown (%)':
        colors = ['red' if v < -10 else 'orange' if v < -5 else 'green' for v in vals]
    ax.barh(vals.index, vals.values, color=colors, alpha=0.7)
    ax.set_title(metric)
    ax.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/backtest_metrics.png', dpi=150)
print(f"✅ Saved: {OUT_DIR}/backtest_metrics.png")

plt.show()

# ══════════════════════════════════════════════════════════════════════
# 8. FINAL P&L SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"KEY TAKEAWAY — Starting with ${INITIAL_CAPITAL:,}")
print("=" * 60)
for name, m in summary.items():
    emoji = "📈" if m['Profit/Loss ($)'] > 0 else "📉"
    print(f"   {emoji} {name:25s}: ${m['Final Value ($)']:>10,.2f}  "
          f"({m['Total Return (%)']:+.1f}%)  "
          f"Sharpe: {m['Sharpe Ratio']:.2f}")