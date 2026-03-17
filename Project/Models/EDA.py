from H_prep import import_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
from H_helpers import get_cwd
import sys
import os
from H_prep import import_data, clean_data

cwd=get_cwd("STAT-587-Final-Project")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

lookup_df = pd.read_csv(cwd / "Project" / "Data" / "stock_lookup_table.csv")
sector_map=lookup_df.set_index('Ticker')['Sector'].to_dict()
idx=pd.IndexSlice

def display_SP500_index_values(DATA):
    fig, ax=plt.subplots(figsize=(16, 7), dpi=100)
    target_data=DATA[0].loc[:, idx["Close", "Index", "^SPX"]]

    ax.plot(target_data.index, target_data.values, 
            color='#2c3e50', linewidth=1.5, alpha=0.9, label='S&P 500 Index')
    ax.axvspan(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-04-15'), 
            color='#e74c3c', alpha=0.15, label='2020 COVID-19 Recession')
    ax.axvspan(pd.Timestamp('2025-02-01'), pd.Timestamp('2025-05-01'), 
            color='#f39c12', alpha=0.15, label='2025 Tariff Shock')

    ax.set_xlabel("Timeline", fontsize=12, labelpad=12)
    ax.set_ylabel("Closing Value", fontsize=12, labelpad=12)
    ax.grid(visible=True, linestyle=':', alpha=0.5, color='gray')
    ax.legend(loc='upper left', frameon=False, fontsize=20)
    plt.show()


def display_SP500_raw_feature_values(DATA):
    fig, ax1=plt.subplots(figsize=(16, 5), dpi=100)
    data_tail=DATA[0].tail(10)
    ax1.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
    ax1.fill_between(DATA[0].tail(10).index, DATA[0].tail(10).loc[:, idx["Low", "Index", "^SPX"]], DATA[0].tail(10).loc[:, idx["High", "Index", "^SPX"]], color="#FFFACD", alpha=0.8, label="High to Low Span")
    ax1.plot(data_tail.index, data_tail.loc[:, idx["Close", "Index", "^SPX"]], color="#66BDFF", marker='o', linestyle='-', alpha=1, label="Close")
    ax1.plot(data_tail.index, data_tail.loc[:, idx["Open", "Index", "^SPX"]], color="#77FF58", marker='o', linestyle='-', alpha=1, label="Open")
    ax1.plot(data_tail.index, data_tail.loc[:, idx["Low", "Index", "^SPX"]], color="black", linestyle='--', alpha=0.5)
    ax1.plot(data_tail.index, data_tail.loc[:, idx["High", "Index", "^SPX"]], color="black", linestyle='--', alpha=0.5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    
    ax2=ax1.twinx()
    ax2.bar(data_tail.index, data_tail.loc[:, idx["Volume", "Index", "^SPX"]], color='gray', alpha=0.3, width=0.2, label="Volume")
    ax2.set_ylim(0, data_tail.loc[:, idx["Volume", "Index", "^SPX"]].max() * 4) 
    ax2.set_ylabel("Volume")
    
    lines1, labels1=ax1.get_legend_handles_labels()
    lines2, labels2=ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title("S&P 500 Price & Volume (Last 10 Days)")
    plt.show()

def display_nonstationary_metric(DATA):
    plt.figure(figsize=(8, 5))
    plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
    plt.plot(DATA[0].tail(100).index, DATA[0].tail(100).loc[:, idx["Close", "Index", "^SPX"]], linestyle='-', alpha=0.5)
    temp_mean=DATA[0].tail(100).loc[:, idx["Close", "Index", "^SPX"]].mean()
    plt.plot(DATA[0].tail(100).index, [temp_mean for _ in range(100)], linestyle='-', alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Closing Value")
    plt.show()

def display_stationary_metric(DATA):
    plt.figure(figsize=(8, 5))
    plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
    plt.plot(DATA[0].tail(100).index, DATA[0].tail(101).loc[:, idx["Close", "Index", "^SPX"]].pct_change().dropna(how="any", axis=0), linestyle='-', alpha=0.5)
    plt.plot(DATA[0].tail(100).index, [0 for _ in range(100)], linestyle='-', alpha=0.5)
    plt.xlabel("Date")
    plt.ylabel("Closing Return Value")
    plt.show()

def display_SP500_return_histogram(DATA):
    spx_returns=DATA[1]
    mu=spx_returns.mean()
    s=spx_returns.std()

    fig, ax = plt.subplots(figsize=(16, 6), dpi=100)
    sns.set_style("white")
    sns.histplot(spx_returns, bins=500, stat="density", color='#66BDFF', alpha=0.4, ax=ax, label='Daily Returns Distribution')
    x = np.linspace(spx_returns.min(), spx_returns.max(), 250)
    p = norm.pdf(x, mu, s)

    plt.plot(x, p, 'r', lw=2, label=f'Normal Dist. ($\mu={mu:.5f}, \sigma={s:.5f}$)')
    plt.axvline(0.0, color='r', linestyle='--', alpha=0.4, linewidth=1, label="Base Line")
    plt.title('Distribution of Index Returns vs. Normal Distribution', fontsize=14)
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def display_classification_counts(DATA):
    spx_returns=DATA[1]
    day_classifier=['Up Day', 'Down Day']
    day_classifications=[]
    day_classifications.append((spx_returns > 0).sum())
    day_classifications.append((spx_returns < 0).sum())

    neutral_days=(spx_returns==0).sum()
    if (neutral_days != 0):
        day_classifications.append(neutral_days)
        day_classifier.append('Neutral Day')

    plt.bar(day_classifier, day_classifications)
    plt.title('Daily Returns Timeline')
    plt.ylabel('Return Classification')

def plot_sector_correlation(X, y, sector_name =None):
    returns_df=X.copy().xs(key='Close PC', axis=1, level=0)
    returns_df.columns=returns_df.columns.droplevel(0)
    
    if (sector_name != None):
        sector_tickers=lookup_df[lookup_df['Sector'] == sector_name]['Ticker']
        sector_returns=returns_df[returns_df.columns.intersection(sector_tickers)]
        sector_corr=sector_returns.corr()
        g = sns.clustermap(sector_corr, 
                        row_cluster=True,
                        col_cluster=True, 
                        annot=False, 
                        cmap='YlGnBu', 
                        vmin=0, vmax=1,
                        figsize=(6, 6))

        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(False)
        g.figure.suptitle(f"Clustered Intra-Sector Correlation: {sector_name}", y=1.02)
        plt.suptitle(f"Sorted Intra-Sector Correlation: {sector_name}", y=0.95)
        plt.show()
    else:
        sector_returns=returns_df.T.groupby(sector_map).mean().T
        sector_correlation=sector_returns.corr()
        sector_correlation.index.name = None
        sector_correlation.columns.name = None
        plt.figure(figsize=(10, 8))
        sns.heatmap(sector_correlation, annot=True, cmap='RdYlGn', center=0, 
                    square=True,       
                    cbar_kws={"shrink": .8})
        plt.title("Correlation of Sector Returns")
        plt.show()

# plot_sector_correlation(X, y, 'Technology')
# plot_sector_correlation('Financial Services', returns_df, lookup_df)
# plot_sector_correlation('Real Estate', returns_df, lookup_df)

def display_volatility(X, y):
    X, y=clean_data(X.copy(), y.copy(), lookback_period=28)
    vol_df=X.xs(key='Close VOL 28', axis=1, level=0)
    vol_df.columns=vol_df.columns.droplevel(0)
    sector_vol_series=vol_df.T.groupby(sector_map).mean().T
    total_average_volatility=vol_df.mean(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(total_average_volatility.index, total_average_volatility, color="black", linewidth=2, label="Total Market")
    sector_vol_series=vol_df.T.groupby(sector_map).mean().T
    for sector in sector_vol_series.columns:
        plt.plot(sector_vol_series.index, sector_vol_series[sector], label=sector, alpha=0.35, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

def display_daily_returns(X, y):
    X, y=clean_data(X.copy(), y.copy(), raw=True)
    ret_df=X.xs(key='Close PC', axis=1, level=0)
    ret_df.columns=ret_df.columns.droplevel(0)
    sector_ret_series=ret_df.T.groupby(sector_map).mean().T

    total_average_ret=ret_df.mean(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(list(total_average_ret.index)[-100:], total_average_ret.iloc[-100:].rolling(window=10).mean(), color="black", linewidth=2, label="Total Market")
    for sector in sector_ret_series.columns:
        plt.plot(list(sector_ret_series.index)[-100:], sector_ret_series[sector].iloc[-100:].rolling(window=10).mean(), label=sector, alpha=0.35, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Daily Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()