import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

cwd=Path.cwd()
for _ in range(5): 
    if cwd.name!="STAT-587-Final-Project":
        cwd=cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

import sys
import os
sys.path.append(os.path.abspath(cwd / "Project" / "Models"))
from H_prep import import_data, clean_data


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 8)

lookup_df = pd.read_csv(cwd / "Project" / "Data" / "stock_lookup_table.csv")

X, y=import_data(corr_level=0, testing=True)

# returns_df=X.xs(key='Close PC', axis=1, level=0)
# returns_df.columns=returns_df.columns.droplevel(0)
# sector_map=lookup_df.set_index('Ticker')['Sector'].to_dict()
# sector_returns = returns_df.T.groupby(sector_map).mean().T
# sector_correlation=sector_returns.corr()
# sector_correlation.index.name = None
# sector_correlation.columns.name = None
# plt.figure(figsize=(10, 8))
# sns.heatmap(sector_correlation, annot=True, cmap='RdYlGn', center=0, 
#             square=True,       
#             cbar_kws={"shrink": .8})
# plt.title("Correlation of Sector Returns")
# plt.savefig(f'../all_sector_correlation_heatmap.png', dpi=600, bbox_inches="tight")
# plt.show()

# def plot_sector_correlation(sector_name, returns_df, lookup_df):
#     sector_tickers=lookup_df[lookup_df['Sector'] == sector_name]['Ticker']
#     sector_returns=returns_df[returns_df.columns.intersection(sector_tickers)]
#     sector_corr=sector_returns.corr()
#     g = sns.clustermap(sector_corr, 
#                     row_cluster=True,
#                     col_cluster=True, 
#                     annot=False, 
#                     cmap='YlGnBu', 
#                     vmin=0, vmax=1,
#                     figsize=(6, 6))

#     g.ax_row_dendrogram.set_visible(False)
#     g.ax_col_dendrogram.set_visible(False)
#     g.figure.suptitle(f"Clustered Intra-Sector Correlation: {sector_name}", y=1.02)

#     plt.suptitle(f"Sorted Intra-Sector Correlation: {sector_name}", y=0.95)
#     plt.savefig(f'../{sector_name}_correlation_heatmap.png', dpi=600, bbox_inches="tight")
#     plt.show()
#     input("wait")

# plot_sector_correlation('Technology', returns_df, lookup_df)
# plot_sector_correlation('Financial Services', returns_df, lookup_df)
# plot_sector_correlation('Real Estate', returns_df, lookup_df)

X, y=clean_data(X, y, lookback_period=28)
vol_df=X.xs(key='Close VOL 28', axis=1, level=0)
vol_df.columns=vol_df.columns.droplevel(0)
sector_map=lookup_df.set_index('Ticker')['Sector'].to_dict()
sector_vol_series=vol_df.T.groupby(sector_map).mean().T

X, y=clean_data(X, y, raw=True)
ret_df=X.xs(key='Close PC', axis=1, level=0)
ret_df.columns=ret_df.columns.droplevel(0)
sector_ret_series=ret_df.T.groupby(sector_map).mean().T
# for sector in sector_vol_series.columns:
#     plt.plot(sector_vol_series.index, sector_vol_series[sector], label=sector)
# plt.title("Sector Volatility Over Time")
# plt.xlabel("Date")
# plt.ylabel("Volatility")
# plt.legend()
# plt.show()

# smoothed_sector_vol = sector_vol_series.rolling(window=40).mean()
# for sector in smoothed_sector_vol.columns:
#     plt.plot(smoothed_sector_vol.index, smoothed_sector_vol[sector], label=sector)
# plt.title("Smoothed Sector Volatility Over Time")
# plt.xlabel("Date")
# plt.ylabel("Volatility")
# plt.legend()
# plt.tight_layout()
# plt.show()

# total_average_volatility=vol_df.mean(axis=1)
# plt.figure(figsize=(10, 6))
# plt.plot(total_average_volatility.index, total_average_volatility, color="black", linewidth=2, label="Total Market")
# sector_vol_series=vol_df.T.groupby(sector_vol_map).mean().T
# for sector in sector_vol_series.columns:
#     plt.plot(sector_vol_series.index, sector_vol_series[sector], label=sector, alpha=0.35, linewidth=1)
# plt.xlabel("Date")
# plt.ylabel("Volatility")
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'Project/Models/results/image_results/SP500FeatureImages/all_sector_rolling_VOL_28.png', dpi=600, bbox_inches="tight")
# plt.show()

total_average_ret=ret_df.mean(axis=1)
plt.figure(figsize=(10, 6))
plt.plot(list(total_average_ret.index)[-100:], total_average_ret.iloc[-100:].rolling(window=10).mean(), color="black", linewidth=2, label="Total Market")
for sector in sector_ret_series.columns:
    plt.plot(list(sector_ret_series.index)[-100:], sector_ret_series[sector].iloc[-100:].rolling(window=10).mean(), label=sector, alpha=0.35, linewidth=1)
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.legend()
plt.tight_layout()
plt.savefig(f'Project/Models/results/image_results/SP500FeatureImages/all_sector_daily_return.png', dpi=600, bbox_inches="tight")
plt.show()