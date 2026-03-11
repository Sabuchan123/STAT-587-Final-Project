from H_prep import import_data
import matplotlib.pyplot as plt
import pandas as pd

DATA=import_data(extra_features=False, corr_level=0)
DATA[0].dropna(how='any', axis=0, inplace=True)

idx=pd.IndexSlice

fig, ax = plt.subplots(figsize=(16, 7), dpi=100)
target_data = DATA[0].loc[:, idx["Close", "Index", "^SPX"]]

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

plt.savefig('Project/Models/results/image_results/SP500IndexValues/sp500_styled_analysis.png', dpi=600, bbox_inches="tight")
plt.show()

# plt.figure(figsize=(16, 5), dpi=100)
# plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# plt.plot(DATA[0].index, DATA[0].loc[:, idx["Close", "Index", "^SPX"]], linestyle='-', alpha=0.5)
# plt.xlabel("Date")
# plt.ylabel("Closing Value")
# plt.savefig('../s&p500_index_values.png', dpi=600, bbox_inches="tight")
# plt.show()

# fig, ax1=plt.subplots(figsize=(16, 5), dpi=100)
# data_tail=DATA[0].tail(10)
# ax1.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# ax1.fill_between(DATA[0].tail(10).index, DATA[0].tail(10).loc[:, idx["Low", "Index", "^SPX"]], DATA[0].tail(10).loc[:, idx["High", "Index", "^SPX"]], color="#FFFACD", alpha=0.8, label="High to Low Span")
# ax1.plot(data_tail.index, data_tail.loc[:, idx["Close", "Index", "^SPX"]], color="#66BDFF", marker='o', linestyle='-', alpha=1, label="Close")
# ax1.plot(data_tail.index, data_tail.loc[:, idx["Open", "Index", "^SPX"]], color="#77FF58", marker='o', linestyle='-', alpha=1, label="Open")
# ax1.plot(data_tail.index, data_tail.loc[:, idx["Low", "Index", "^SPX"]], color="black", linestyle='--', alpha=0.5)
# ax1.plot(data_tail.index, data_tail.loc[:, idx["High", "Index", "^SPX"]], color="black", linestyle='--', alpha=0.5)
# ax1.set_xlabel("Value")
# ax1.set_ylabel("Date")

# ax2=ax1.twinx()
# ax2.bar(data_tail.index, data_tail.loc[:, idx["Volume", "Index", "^SPX"]], color='gray', alpha=0.3, width=0.2, label="Volume")
# ax2.set_ylim(0, data_tail.loc[:, idx["Volume", "Index", "^SPX"]].max() * 4) 
# ax2.set_ylabel("Volume")

# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
# plt.title("S&P 500 Price & Volume (Last 10 Days)")
# plt.savefig('../s&p500_index_values_data_display.png', dpi=600, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# plt.plot(DATA[0].tail(100).index, DATA[0].tail(100).loc[:, idx["Close", "Index", "^SPX"]], linestyle='-', alpha=0.5)
# temp_mean=DATA[0].tail(100).loc[:, idx["Close", "Index", "^SPX"]].mean()
# plt.plot(DATA[0].tail(100).index, [temp_mean for _ in range(100)], linestyle='-', alpha=0.5)
# plt.xlabel("Date")
# plt.ylabel("Closing Value")
# plt.savefig('../s&p500_index_values_past_100_days.png', dpi=600, bbox_inches="tight")
# plt.show()

# plt.figure(figsize=(8, 5))
# plt.grid(visible=True, linestyle='--', alpha=0.5, color='gray')
# plt.plot(DATA[0].tail(100).index, DATA[0].tail(101).loc[:, idx["Close", "Index", "^SPX"]].pct_change().dropna(how="any", axis=0), linestyle='-', alpha=0.5)
# plt.plot(DATA[0].tail(100).index, [0 for _ in range(100)], linestyle='-', alpha=0.5)
# plt.xlabel("Date")
# plt.ylabel("Closing Return Value")
# plt.savefig('../s&p500_index_values_past_100_days_close_pc.png', dpi=600, bbox_inches="tight")
# plt.show()