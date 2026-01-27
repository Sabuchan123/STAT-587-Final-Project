#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from pathlib import Path

DATA=pd.read_csv(Path('../bin/input_data.csv'), header=[0, 1, 2], index_col=0, parse_dates=True)
print(DATA.head())