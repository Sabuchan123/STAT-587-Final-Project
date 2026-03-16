from H_prep import import_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
from H_helpers import get_cwd

cwd=get_cwd("STAT-587-Final-Project")

DATA=import_data(corr_level=0)
DATA[0].dropna(how='any', axis=0, inplace=True)

idx=pd.IndexSlice

# fig, ax=plt.subplots(figsize=(16, 7), dpi=100)
# target_data=DATA[0].loc[:, idx["Close", "Index", "^SPX"]]

# ax.plot(target_data.index, target_data.values, 
#         color='#2c3e50', linewidth=1.5, alpha=0.9, label='S&P 500 Index')
# ax.axvspan(pd.Timestamp('2020-02-15'), pd.Timestamp('2020-04-15'), 
#            color='#e74c3c', alpha=0.15, label='2020 COVID-19 Recession')
# ax.axvspan(pd.Timestamp('2025-02-01'), pd.Timestamp('2025-05-01'), 
#            color='#f39c12', alpha=0.15, label='2025 Tariff Shock')

# ax.set_xlabel("Timeline", fontsize=12, labelpad=12)
# ax.set_ylabel("Closing Value", fontsize=12, labelpad=12)
# ax.grid(visible=True, linestyle=':', alpha=0.5, color='gray')
# ax.legend(loc='upper left', frameon=False, fontsize=20)

# plt.savefig('Project/Models/results/image_results/SP500IndexValues/sp500_styled_analysis.png', dpi=600, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(16, 5), dpi=100)
# plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# plt.plot(DATA[0].index, DATA[0].loc[:, idx["Close", "Index", "^SPX"]], linestyle='-', alpha=0.5)
# plt.xlabel("Date")
# plt.ylabel("Closing Value")
# plt.savefig('../sp500_index_values.png', dpi=600, bbox_inches="tight")
# plt.show()

# fig, ax1=plt.subplots(figsize=(16, 5), dpi=100)
# data_tail=DATA[0].tail(10)
# ax1.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# ax1.fill_between(DATA[0].tail(10).index, DATA[0].tail(10).loc[:, idx["Low", "Index", "^SPX"]], DATA[0].tail(10).loc[:, idx["High", "Index", "^SPX"]], color="#FFFACD", alpha=0.8, label="High to Low Span")
# ax1.plot(data_tail.index, data_tail.loc[:, idx["Close", "Index", "^SPX"]], color="#66BDFF", marker='o', linestyle='-', alpha=1, label="Close")
# ax1.plot(data_tail.index, data_tail.loc[:, idx["Open", "Index", "^SPX"]], color="#77FF58", marker='o', linestyle='-', alpha=1, label="Open")
# ax1.plot(data_tail.index, data_tail.loc[:, idx["Low", "Index", "^SPX"]], color="black", linestyle='--', alpha=0.5)
# ax1.plot(data_tail.index, data_tail.loc[:, idx["High", "Index", "^SPX"]], color="black", linestyle='--', alpha=0.5)
# ax1.set_xlabel("Date")
# ax1.set_ylabel("Value")

# ax2=ax1.twinx()
# ax2.bar(data_tail.index, data_tail.loc[:, idx["Volume", "Index", "^SPX"]], color='gray', alpha=0.3, width=0.2, label="Volume")
# ax2.set_ylim(0, data_tail.loc[:, idx["Volume", "Index", "^SPX"]].max() * 4) 
# ax2.set_ylabel("Volume")

# lines1, labels1=ax1.get_legend_handles_labels()
# lines2, labels2=ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
# plt.title("S&P 500 Price & Volume (Last 10 Days)")
# plt.savefig('../sp500_index_values_data_display.png', dpi=600, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# plt.plot(DATA[0].tail(100).index, DATA[0].tail(100).loc[:, idx["Close", "Index", "^SPX"]], linestyle='-', alpha=0.5)
# temp_mean=DATA[0].tail(100).loc[:, idx["Close", "Index", "^SPX"]].mean()
# plt.plot(DATA[0].tail(100).index, [temp_mean for _ in range(100)], linestyle='-', alpha=0.5)
# plt.xlabel("Date")
# plt.ylabel("Closing Value")
# plt.savefig('../sp500_index_values_past_100_days.png', dpi=600, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# plt.plot(DATA[0].tail(100).index, DATA[0].tail(101).loc[:, idx["Close", "Index", "^SPX"]].pct_change().dropna(how="any", axis=0), linestyle='-', alpha=0.5)
# plt.plot(DATA[0].tail(100).index, [0 for _ in range(100)], linestyle='-', alpha=0.5)
# plt.xlabel("Date")
# plt.ylabel("Closing Return Value")
# plt.savefig('../sp500_index_values_past_100_days_close_pc.png', dpi=600, bbox_inches="tight")
# plt.show()

# df=pd.read_csv(cwd / 'Project' / 'Models' / 'results' / 'results.csv')
# df['test_accuracy_edge']=df['test_split_accuracy'] - 0.5
# plot_df=df[['label', 'utility_score', 'test_accuracy_edge']].copy()
# plot_df.columns = ['Model', 'Utility Score (Obj)', 'Test Accuracy Edge (Acc - 0.5)']
# plot_df.set_index('Model', inplace=True)

# ax=plot_df.plot(kind='bar', figsize=(10, 6), width=0.8, color=['#e74c3c', '#3498db'])

# plt.title('Comparison of Utility Score vs. Test Accuracy', fontsize=14)
# plt.ylabel('Value', fontsize=12)
# plt.xlabel('Model Configuration', fontsize=12)
# plt.xticks(rotation=0) # Keep model names horizontal for readability
# plt.axhline(0, color='black', linewidth=1.0, linestyle='-') # Baseline 0
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.legend(loc='best')

# plt.tight_layout()
# plt.savefig(cwd / 'Project' / 'Models' / 'results' / 'model_comparison_results.png', dpi=300)

spx_returns=DATA[1]
# mu=spx_returns.mean()
# s=spx_returns.std()
# kurt=spx_returns.kurt()
# skew=spx_returns.skew()
# print(mu)
# print(s)
# print(kurt)
# print(skew)

# fig, ax = plt.subplots(figsize=(16, 6), dpi=100)
# sns.set_style("white")
# sns.histplot(spx_returns, bins=500, stat="density", color='#66BDFF', alpha=0.4, ax=ax, label='Daily Returns Distribution')
# x = np.linspace(spx_returns.min(), spx_returns.max(), 250)
# p = norm.pdf(x, mu, s)

# plt.plot(x, p, 'r', lw=2, label=f'Normal Dist. ($\mu={mu:.5f}, \sigma={s:.5f}$)')
# plt.axvline(0.0, color='r', linestyle='--', alpha=0.4, linewidth=1, label="Base Line")
# plt.title('Distribution of Index Returns vs. Normal Distribution', fontsize=14)
# plt.xlabel('Daily Return')
# plt.ylabel('Density')
# plt.legend()
# plt.savefig('Project/Models/results/image_results/SP500IndexValues/sp500_histogram.png', dpi=600, bbox_inches='tight')
# plt.show()

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
plt.savefig(cwd / 'Project' / 'Models' / 'results' / 'image_results' / 'SP500IndexValues' / 'model_comparison_results.png', dpi=300)