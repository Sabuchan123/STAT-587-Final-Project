#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from pathlib import Path

cwd=Path.cwd()
while (cwd.name!="STAT-587-Final-Project"): cwd=cwd.parent

# Hyperparameters
lag=[1, 3, 7, 14]
ema_windows=[7, 14, 21]
vol_windows=[14, 28]
max_min_windows=[7, 21]
rol_VWAP_windows=[7, 14, 21]
rol_zscore_windows=[7, 14, 21]

# Day of the week lists for Day of the Week analysis
Monday=[]
Tuesday=[]
Wednesday=[]
Thursday=[]
Friday=[]

idx = pd.IndexSlice

DATA=pd.read_csv(cwd / "bin" / "total_data.csv", header=[0, 1, 2], index_col=0, parse_dates=True)

MODIFIED_DATA=DATA.loc[:, idx[['Close', 'Open', 'High', 'Low'], :, :]].copy().pct_change().rename(columns={metric: f"{metric} PC" for metric in ['Close', 'Open', 'High', 'Low']}, level=0).iloc[1:]

MODIFIED_DATA=pd.concat([MODIFIED_DATA, DATA.loc[:, idx[['Close', 'Open', 'High', 'Low', 'Volume'], :, :]].copy().iloc[1:]], axis=1)

for metric in ['Close PC', 'Open PC', 'High PC', 'Low PC']:
    for lag_period in lag:
        MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx[metric, :, :]].shift(lag_period).rename(columns={metric: f"{metric} Lag {lag_period}"}, level=0)], axis=1)


# TODO Utilize Volatility to get Z-Score rows
for metric in ['Close', 'Open', 'High', 'Low']:
    for ema_window in ema_windows: 
        MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx[metric, :, :]].ewm(span=ema_window, adjust=False).mean().rename(columns={metric: f"{metric} EMA {ema_window}"}, level=0)], axis=1)
    for vol_window in vol_windows:
        MODIFIED_DATA=pd.concat([MODIFIED_DATA, MODIFIED_DATA.loc[:, idx[metric, :, :]].rolling(window=vol_window).std().rename(columns={metric: f"{metric} VOL {vol_window}"}, level=0)], axis=1)

print(MODIFIED_DATA.columns.levels[0])
print(MODIFIED_DATA["Close EMA 21"]["Stocks"].iloc[0:100])
print(MODIFIED_DATA.iloc[100:150])