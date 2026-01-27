#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from pathlib import Path

# Hyperparameters
lag=[1, 3, 7, 14]
ma_window=[7, 14, 21]
vol_window=[14, 28]
max_min_window=[7, 21]
rol_VWAP_window=[7, 14, 21]
rol_zscore_window=[7, 14, 21]

# Day of the week lists for Day of the Week analysis
Monday=[]
Tuesday=[]
Wednesday=[]
Thursday=[]
Friday=[]

idx = pd.IndexSlice

DATA=pd.read_csv(Path('../bin/total_data.csv'), header=[0, 1, 2], index_col=0, parse_dates=True)

MODIFIED_DATA=DATA.loc[:, idx[['Close', 'Open', 'High', 'Low'], :, :]].copy().pct_change().rename(columns={metric: f"{metric}_pc" for metric in ['Close', 'Open', 'High', 'Low']}, level=0).iloc[1:]

MODIFIED_DATA=pd.concat([MODIFIED_DATA, DATA.loc[:, idx['Volume', :, :]].copy().iloc[1:]], axis=1)

# for metric in ['Close', 'Open', 'High', 'Low', 'Volume']:
#     for lag_period in lag:
#         MODIFIED_DATA.loc[:, idx[f'Lag_{lag_period}', :, :]] = MODIFIED_DATA.loc[:, idx[metric, :, :]].shift(lag_period)

print(MODIFIED_DATA.head())