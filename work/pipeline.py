import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs('../data/processed', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD RAW CSVs
# ══════════════════════════════════════════════════════════════════════
FILES = {
    'AAPL': '../data/Apple.csv',
    'MSFT': '../data/MSFT.csv',
    'GOOGL': '../data/google.csv',
    'AMZN': '../data/Amazon.csv',
    'NVDA': '../data/NVIDIA.csv',
}

def load(ticker, path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').set_index('Date')
    df.columns = [c.strip() for c in df.columns]
    df = df[['Open','High','Low','Close','Adj Close','Volume']].copy()
    df.columns = ['Open','High','Low','Close','AdjClose','Volume']
    # Remove commas and convert all columns to numeric
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

stocks = {t: load(t, p) for t, p in FILES.items()}
print("✅ Step 1: Raw data loaded")

# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (daily indicators)
# ══════════════════════════════════════════════════════════════════════
def add_indicators(df):
    c = df['AdjClose']
    v = df['Volume']

    for w in [5, 10, 20, 50]:
        df[f'SMA_{w}'] = c.rolling(w).mean()
    for w in [10, 20]:
        df[f'EMA_{w}'] = c.ewm(span=w, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['BB_Upper']  = sma20 + 2 * std20
    df['BB_Lower']  = sma20 - 2 * std20
    df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / sma20

    df['Vol_SMA_10']     = v.rolling(10).mean()
    df['Vol_Change_Pct'] = v.pct_change(fill_method=None) * 100
    df['Daily_Return']   = c.pct_change(fill_method=None) * 100
    return df

for t in stocks:
    stocks[t] = add_indicators(stocks[t])
print("✅ Step 2: Technical indicators added")

# ══════════════════════════════════════════════════════════════════════
# 3. RESAMPLE TO WEEKLY
# ══════════════════════════════════════════════════════════════════════
def to_weekly(df, ticker):
    w = pd.DataFrame()
    w['Open']   = df['Open'].resample('W-FRI').first()
    w['High']   = df['High'].resample('W-FRI').max()
    w['Low']    = df['Low'].resample('W-FRI').min()
    w['Close']  = df['AdjClose'].resample('W-FRI').last()
    w['Volume'] = df['Volume'].resample('W-FRI').sum()

    ind_cols = [c for c in df.columns if c not in
                ['Open','High','Low','Close','AdjClose','Volume','Daily_Return']]
    for col in ind_cols:
        w[col] = df[col].resample('W-FRI').mean()

    w['Weekly_Return'] = w['Close'].pct_change(fill_method=None) * 100
    for lag in [1, 2, 3]:
        w[f'Return_Lag{lag}'] = w['Weekly_Return'].shift(lag)

    w['Ticker'] = ticker
    return w

weekly = {t: to_weekly(stocks[t], t) for t in stocks}
print("✅ Step 3: Resampled to weekly")

# ══════════════════════════════════════════════════════════════════════
# 4. ALIGN DATES & COMBINE
# ══════════════════════════════════════════════════════════════════════
common_idx = weekly['AAPL'].index
for t in weekly:
    common_idx = common_idx.intersection(weekly[t].index)
for t in weekly:
    weekly[t] = weekly[t].loc[common_idx]

all_df = pd.concat(weekly.values()).sort_index()

sp500_return = all_df.groupby(level=0)['Weekly_Return'].mean().rename('SP500_Proxy')
all_df = all_df.join(sp500_return, how='left')
all_df['Excess_Return'] = all_df['Weekly_Return'] - all_df['SP500_Proxy']
print("✅ Step 4: Dates aligned and combined")

# ══════════════════════════════════════════════════════════════════════
# 5. WEEKLY RANKING LABELS
# ══════════════════════════════════════════════════════════════════════
pivot   = all_df.pivot_table(index=all_df.index, columns='Ticker', values='Weekly_Return')
rank_df = pivot.rank(axis=1, ascending=False).round(0).astype('Int64')
rank_df.columns = [f'Rank_{c}' for c in rank_df.columns]

all_df  = all_df.reset_index().rename(columns={'index':'Date'})
rank_df = rank_df.reset_index().rename(columns={'Date':'Date'})
all_df  = all_df.merge(rank_df, on='Date', how='left')
all_df['Weekly_Rank'] = all_df.apply(lambda r: r[f"Rank_{r['Ticker']}"], axis=1)

all_df = all_df.dropna(subset=['Weekly_Return','RSI_14','MACD','Weekly_Rank'])
all_df = all_df.sort_values(['Date','Ticker']).reset_index(drop=True)
print("✅ Step 5: Weekly ranking labels created")

# ══════════════════════════════════════════════════════════════════════
# 6. TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════
train = all_df[all_df['Date'] <= '2021-12-31'].copy()
val   = all_df[(all_df['Date'] >= '2022-01-01') & (all_df['Date'] <= '2023-12-31')].copy()
test  = all_df[all_df['Date'] >= '2024-01-01'].copy()
print("✅ Step 6: Train/Val/Test split done")
print(f"   Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")

# ══════════════════════════════════════════════════════════════════════
# 7. FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════
exclude_cols = ['Date','Ticker','Weekly_Rank',
                'Rank_AAPL','Rank_AMZN','Rank_GOOGL','Rank_MSFT','Rank_NVDA']
feature_cols = [c for c in all_df.columns if c not in exclude_cols]

scaler = StandardScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
val[feature_cols]   = scaler.transform(val[feature_cols])
test[feature_cols]  = scaler.transform(test[feature_cols])

joblib.dump(scaler, '../data/processed/scaler.pkl')
print("✅ Step 7: Features scaled (scaler saved)")

# ══════════════════════════════════════════════════════════════════════
# 8. SAVE FINAL FILES
# ══════════════════════════════════════════════════════════════════════
train.to_csv('../data/processed/train.csv', index=False)
val.to_csv('../data/processed/val.csv',     index=False)
test.to_csv('../data/processed/test.csv',   index=False)

print("\n✅ Pipeline complete! Files saved to data/processed/")
print("   train.csv | val.csv | test.csv | scaler.pkl")
print(f"\n   Total features : {len(feature_cols)}")
print(f"   Feature list   : {feature_cols}")