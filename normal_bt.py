import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import vectorbt as vbt
import time 
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

symbol = "BTCUSDT"
timeframe = "5m"
startdate = "2021-04-01"

# TODO 攞幣的ohlc 
# df = vbt.CCXTData.download_symbol(symbol=symbol, exchange="binance", start=startdate, timeframe=timeframe)
# df.columns =[ 'open', 'high', 'low', 'close', 'volume']# [ f'{i}_open', f'{i}_high', f'{i}_low', f'{i}_close', 'volume']
# df.index.names =['date']
# # data = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
# df.to_csv(f"../data/{symbol}-{timeframe}.csv")

# TODO 攞完一次 save低後，不想再由API攞，係save低的DATA 到攞
df = pd.read_csv(f"../data/{symbol}-{timeframe}.csv")
df = df.reset_index()
from datetime import datetime
print(type(df['date'][0]))

# TODO 融合2個DATA：df = 幣ohlc, ndf = glassnode DATA, 靠timestamp（t）去combin 2 組DATA
df["date"] = pd.to_datetime(df['date'])
df['t'] = df['date'].apply(lambda x: datetime.timestamp(x))
print(df)
sym = symbol.replace("USDT","")

# TODO your secondary data 
ndf = pd.read_parquet(f"netflowdata/bidask_{symbol}-5m.parquet")
ndf = ndf.reset_index()
ndf['date'] = pd.to_datetime(ndf['timestamp'])
ndf.rename({'timestamp': 't'}, axis=1, inplace=True)
ndf.rename({'ratio': 'v'}, axis=1, inplace=True)

df = pd.merge(df, ndf[['t', 'v']],how='inner',on='t')

print(df)
df['chg'] = df['close'].pct_change()


sharpe_tr = {
    "1m": 24*60,
    "5m": 24*12,
    "15m": 24*4, 
    "30m": 24*2, 
    "1h": 24,
    "1d": 1,
}


def backtesting(window, threshold):
    df['ma'] = df['v'].rolling(window).mean()
    # df['per'] = df['v'] / df['ma'] - 1
    
    df['sd'] = df['v'].rolling(window).std()
    df['z'] = ( df['v'] - df['ma'] ) / df['sd']


    # for i in range(len(df)):
    #     if df.loc[i,'z'] > threshold:
    #         df.loc[i,'pos'] = 1
    #     elif df.loc[i,'z'] < -threshold:
    #         df.loc[i, 'pos'] = -1
    #     else:
    #         df.loc[i, 'pos'] = 0

    df['pos'] = np.where(df['z'] > threshold, -1,np.where(df['z'] < -threshold,1,0)) 
    
    df['pos_t-1'] = df['pos'].shift(1)
    df['trade'] = abs(df['pos'] - df['pos_t-1']) #手續費睇倉位轉變
    df['pnl'] = df['pos_t-1'] * df['chg'] - df['trade'] * 0.05/100 #收5滴
    df['cumu'] = df['pnl'].cumsum()
    df['dd'] = df['cumu'].cummax() - df['cumu']

    df['bnh_pnl'] = df['chg']
    df.loc[0:window,'bnh_pnl'] = 0
    # df['bnh_pnl'][:window] = 0
    # print(df['bnh_pnl'][:window])
    df['bnh_cumu'] = df['chg'].cumsum()

    annual_return = round(df['pnl'].mean() * 365 , 3)
    sharpe = round(df['pnl'].mean() / df['pnl'].std() * np.sqrt(365*sharpe_tr[timeframe]), 3)
    mdd = round(df['dd'].max(), 3)
    calmar = round(annual_return / mdd, 3)

    bnh_sharpe = df['bnh_pnl'].mean() / df['bnh_pnl'].std()*np.sqrt(365*sharpe_tr[timeframe]),3

    average_return = df.loc[window:len(df),'pnl'].mean()
    return_sd = df.loc[window:len(df),'pnl'].std()
    prescise_sharpe = average_return / return_sd *np.sqrt(365*sharpe_tr[timeframe])
    # print(df)
    # print(window, threshold, 'annual return', annual_return, 'sharpe', sharpe, 'mdd', mdd, 'calmar', calmar)

    return pd.Series([window, threshold, sharpe, round(df['cumu'].iloc[-1], 3)], index=['window', 'threshold', 'sharpe', 'cumu'])

window_list = np.arange(30,1000,10) 
threshold_list = np.arange(0, 0.4, 0.01)

### create a new dataframe
result_df = pd.DataFrame(columns=['window', 'threshold', 'sharpe', 'cumu'])

# ##### Optimization 
for window in window_list:
    for threshold in threshold_list:
        result_df = result_df.append(backtesting(window, threshold), ignore_index=True)


def plotSharpetable(sharpe_df:pd.DataFrame):
    sharpe_df = sharpe_df.sort_values(by='sharpe',ascending=False)
    print(sharpe_df)
    # Create the subplot with two heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    data_table = sharpe_df.pivot(index='window',columns='threshold',values='sharpe')
    sns.heatmap(data_table, ax=axs[0], annot=True, cmap='Greens',fmt='g')
    axs[0].set_title("sharpe")

    # Plot the Sharpe heatmap on the right side
    cumu_df = sharpe_df.sort_values(by='cumu',ascending=False)
    print(cumu_df)
    cumu_table = cumu_df.pivot(index='window',columns='threshold',values='cumu')
    sns.heatmap(cumu_table, ax=axs[1], annot=True, cmap='Greens',fmt='g')
    axs[1].set_title("Cumu Result")

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.05, left=0.05, right=1, top=0.95)

    # Display the plot
    plt.show()

plotSharpetable(result_df)

##### Backtest 
# parameters
# window = 540
# threshold = 0.6
# x = backtesting(window, threshold)
# print(x)
# fig = px.line(df, x='date', y=['cumu','dd','bnh_cumu'], title='strategy')
# fig.show()


# # # TODO plot D trade 係邊到開倉，但係淨係有開倉的位置
# import plotly.graph_objects as go

# candlestick = go.Candlestick(
#                             x= df['date'],
#                             open=df['open'],
#                             high=df['high'],
#                             low=df['low'],
#                             close=df['close']
#                             )

# buyplot= df.loc[df['trade'] != 0, ['close', 'date']]
# print(buyplot)


# fig = go.Figure(data=[candlestick,
# go.Scatter(x=buyplot['date'], y= buyplot['close'], mode= 'markers',marker=dict(size=8, color='#00ff00'),),
#  go.Scatter(x=df["date"], y= df['cumu']),
#  go.Scatter(x=df["date"], y= df['bnh_cumu']),
# #  go.Scatter(x=lowdeviate["timestamp"], y= lowdeviate["close"], mode= 'markers',marker=dict(size=6, color='#ff8f78'),),

# ])

# fig.show()

