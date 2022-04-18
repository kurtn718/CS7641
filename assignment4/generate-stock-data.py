import pandas as pd
import pandas_ta as ta

#df = pd.DataFrame()
#df.ta.indicators()

# Daily data-frame

# 8, 21 Day EMA - SPX
# 21 Day EMA - TLT
# 21 DAy EMA - GLD
# 21 Day EMA - VXX
# Day of Week (1-5)
# Open % Gain
# 12pm % Gain
#  3pm % Gain on VXX, GLD, TLT, SPX
# Open Gap fill % (0-100)
# Yesterday Hi
# Yesterday Lo
# Yesterday Close
# Intraday - 50 SMA-%
# CCI
# Above Cloud
# RSI

spx_daily_df = pd.read_csv('SPX-daily.csv')
spx_daily_df['datetime']= pd.to_datetime(spx_daily_df['datetime'])
spx_daily_df = spx_daily_df.set_index(['datetime'])
spx_ema_21 = spx_daily_df.ta.ema(21).rename("SPX_EMA_21")
spx_ema_8 = spx_daily_df.ta.ema(8).rename("SPX_EMA_8")
spx_close = spx_daily_df['close'].rename("SPX_Close")

tlt_daily_df = pd.read_csv('TLT-daily.csv')
tlt_daily_df['datetime']= pd.to_datetime(tlt_daily_df['datetime'])
tlt_daily_df = tlt_daily_df.set_index(['datetime'])
tlt_ema_21 = tlt_daily_df.ta.ema(21).rename("TLT_EMA_21")
tlt_close = tlt_daily_df['close'].rename("TLT_Close")

gld_daily_df = pd.read_csv('TLT-daily.csv')
gld_daily_df['datetime']= pd.to_datetime(gld_daily_df['datetime'])
gld_daily_df = gld_daily_df.set_index(['datetime'])
gld_ema_21 = gld_daily_df.ta.ema(21).rename("GLD_EMA_21")
gld_close = gld_daily_df['close'].rename("GLD_Close")

# TODO Get this piece of date
#vxx_daily_df = pd.read_csv('VXX-daily.csv')
#vxx_daily_df = vxx_daily_df.set_index(['datetime'])
#vxx_ema_21 = vxx_daily_df.ta.ema(21).rename("VXX_EMA_21")
#vxx_close = vxx_daily_df['close'].rename("VXX_Close")

final_df = pd.concat([spx_close,spx_ema_8,spx_ema_21,tlt_close,tlt_ema_21,gld_close,gld_ema_21],axis=1).dropna()
print(final_df.columns)

spx_intraday_df = pd.read_csv('SPX-30.csv')
spx_intraday_df['datetime']= pd.to_datetime(spx_intraday_df['datetime'])
spx_intraday_df['time'] = spx_intraday_df['datetime'].dt.time
spx_intraday_df = spx_intraday_df.set_index(['datetime'])

spx_intra_sma_50_df = spx_intraday_df.ta.sma(50).rename("SPX_ID_50_SMA")
spx_intra_cci_df = spx_intraday_df.ta.cci(30).rename("SPX_ID_CCI")
spx_intraday_close = spx_intraday_df['close'].rename("SPX_ID_CLOSE")
spx_intraday_open = spx_intraday_df['open'].rename("SPX_ID_OPEN")

tick_df = pd.read_csv('TICK.csv')
tick_df['datetime']= pd.to_datetime(tick_df['datetime'])
tick_df = tick_df.set_index(['datetime'])
tick_df = tick_df['close'].rename("TICK")

vxx_df = pd.read_csv('VXX.csv')
vxx_df['datetime']= pd.to_datetime(vxx_df['datetime'])
vxx_df = vxx_df.set_index(['datetime'])
vxx_intra_sma_50_df = vxx_df.ta.sma(50).rename("VXX_ID_50_SMA")
vxx_df = vxx_df['close'].rename("VXX_ID")

spx_intraday_df = pd.concat([spx_intraday_open, spx_intraday_close,spx_intra_cci_df,spx_intra_sma_50_df,tick_df,vxx_df,vxx_intra_sma_50_df],axis=1).dropna()
print(spx_intraday_df)

spx_intraday_df["SPX_ID_PREV_CLOSE"] = spx_intraday_df.shift(1)["SPX_ID_CLOSE"]
spx_intraday_df["SPX_ID_OPEN_GAP"] = spx_intraday_df["SPX_ID_OPEN"] - spx_intraday_df["SPX_ID_PREV_CLOSE"]
print(spx_intraday_df)

#spx_intraday_df['just_date'] = spx_intraday_df.index.dt.date
#spx_intraday_df['just_time'] = d['Dates'].dt.time

#spx_intraday_df = spx_intraday_df.between_time('13:30', '21:00')
#prices = spx_intraday_df.groupby(pd.Grouper(freq='D')).agg({'Open':'first', 'Close':'last'})
#print(prices)

# Schedule
    # Finalize dataset today
    # Code to plot learning curve

# Wednesday

# Thursday

# Friday

# Sat

# Sunday -- done
