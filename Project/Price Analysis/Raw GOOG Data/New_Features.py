import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as pdr
from pandas_datareader import data
import seaborn as sns
import matplotlib.pyplot as plt
import bs4 as bs
import requests
import tensorflow as tf
import urllib.request, json
from IPython.display import clear_output
from scipy.stats import mstats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV, validation_curve, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from sklearn.model_selection import GridSearchCV
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

all_data = pd.read_csv("GOOG_full_1hour_adjsplitdiv.csv", names=["Date", "Open", "High", "Low", "Close", "Volume"])
#print(all_data)

tickers = ['GOOG']
##test_data = pd.DataFrame()
#"Simple Moving Average" Technical Indicator
yf.pdr_override()

##for i in tickers:
##    #make API call using get_data_yahoo
##    
##test_data = pdr.get_data_yahoo(i, start = dt.datetime(2019,1,1), end = dt.datetime(2019,3,30))
##    #adding another column to differentiate the tickers


all_data['symbol'] = 'GOOG'
##all_data = pd.concat([all_data, test_data])


#Creating Return column
all_data['return'] = all_data.groupby('symbol')['Close'].pct_change()




#takes continual 5-day averages and 15-day averages throughout the whole "all_data" dataframe

all_data['SMA_5'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 5).mean())
all_data['SMA_15'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 15).mean())

#takes the ratio of the 15-day average to the 5-day average (if ratio < 1, that means that the stock is above average recently (you should sell soon since you already have a profit)

all_data['SMA_ratio'] = all_data['SMA_15'] / all_data['SMA_5']

#Plotting
#just copy from matplotlib and change parameters as necessary

##start = dt.datetime.strptime('1/1/2019', '%m/%d/%Y')
##end = dt.datetime.strptime('12/31/2019', '%m/%d/%Y')
##sns.set()
##
##fig = plt.figure(facecolor = 'white', figsize = (20,10))
##
##ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
##ax0.plot(all_data[all_data.symbol=='GOOG'].loc[start:end,['Close','SMA_5','SMA_15']])
##ax0.set_facecolor('ghostwhite')
##ax0.legend(['Close','SMA_5','SMA_15'],ncol=3, loc = 'upper left', fontsize = 15)
##plt.title("Google Stock Price, Slow and Fast Moving Average", fontsize = 20)
##
##ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
##ax1.plot(all_data[all_data.symbol=='GOOG'].loc[start:end,['SMA_ratio']], color = 'blue')
##ax1.legend(['SMA_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
##ax1.set_facecolor('silver')
##plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
##plt.show()


#Wilder's smoothing
def Wilder(data, periods):
    start = np.where(~np.isnan(data))[0][0] #Check if nans present in beginning
    wilder = np.array([np.nan]*len(data))
    wilder[start+periods-1] = data[start:(start+periods)].mean() #Simple Moving Average
    for i in range(start+periods,len(data)):
        wilder[i] = (wilder[i-1]*(periods-1) + data[i])/periods #Wilder Smoothing

    return(wilder)

#"Average True Range" Technical Indicator
print('break') 
all_data['prev_close'] = all_data.groupby('symbol')['Close'].shift(1)
all_data['TR'] = np.maximum((all_data['High'] - all_data['Low']), 
                     np.maximum(abs(all_data['High'] - all_data['prev_close']), 
                     abs(all_data['prev_close'] - all_data['Low'])))
for i in all_data['symbol'].unique():
    TR_data = all_data[all_data.symbol == i].copy()
    print(TR_data)
    all_data.loc[all_data.symbol==i,'ATR_5'] = Wilder(TR_data['TR'], 5)
    all_data.loc[all_data.symbol==i,'ATR_15'] = Wilder(TR_data['TR'], 15)

all_data['ATR_Ratio'] = all_data['ATR_5'] / all_data['ATR_15']


#why doesn't this print??????????


#"Average Directional Index (ADX)" Technical Indicator
all_data['prev_high'] = all_data.groupby('symbol')['High'].shift(1)
all_data['prev_low'] = all_data.groupby('symbol')['Low'].shift(1)

all_data['+DM'] = np.where(~np.isnan(all_data.prev_high),
                           np.where((all_data['High'] > all_data['prev_high']) & 
         (((all_data['High'] - all_data['prev_high']) > (all_data['prev_low'] - all_data['Low']))), 
                                                                  all_data['High'] - all_data['prev_high'], 
                                                                  0),np.nan)

all_data['-DM'] = np.where(~np.isnan(all_data.prev_low),
                           np.where((all_data['prev_low'] > all_data['Low']) & 
         (((all_data['prev_low'] - all_data['Low']) > (all_data['High'] - all_data['prev_high']))), 
                                    all_data['prev_low'] - all_data['Low'], 
                                    0),np.nan)

for i in all_data['symbol'].unique():
    ADX_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'+DM_5'] = Wilder(ADX_data['+DM'], 5)
    all_data.loc[all_data.symbol==i,'-DM_5'] = Wilder(ADX_data['-DM'], 5)
    all_data.loc[all_data.symbol==i,'+DM_15'] = Wilder(ADX_data['+DM'], 15)
    all_data.loc[all_data.symbol==i,'-DM_15'] = Wilder(ADX_data['-DM'], 15)

all_data['+DI_5'] = (all_data['+DM_5']/all_data['ATR_5'])*100
all_data['-DI_5'] = (all_data['-DM_5']/all_data['ATR_5'])*100
all_data['+DI_15'] = (all_data['+DM_15']/all_data['ATR_15'])*100
all_data['-DI_15'] = (all_data['-DM_15']/all_data['ATR_15'])*100

all_data['DX_5'] = (np.round(abs(all_data['+DI_5'] - all_data['-DI_5'])/(all_data['+DI_5'] + all_data['-DI_5']) * 100))

all_data['DX_15'] = (np.round(abs(all_data['+DI_15'] - all_data['-DI_15'])/(all_data['+DI_15'] + all_data['-DI_15']) * 100))

for i in all_data['symbol'].unique():
    ADX_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'ADX_5'] = Wilder(ADX_data['DX_5'], 5)
    all_data.loc[all_data.symbol==i,'ADX_15'] = Wilder(ADX_data['DX_15'], 15)





#populate the "Outcome" column on a dataframe
all_data['outcome'] = ''

#outcome has to have the same length as all other 
outcome = ['N/A']
#close is the 4th column in the all_data dataframe
for i in range(1, len(all_data[all_data.columns[[3]]])):
    if all_data.iloc[i, 3] - all_data.iloc[i-1, 3] >= 0:
        outcome.append('B')
    else:
        outcome.append('S')
all_data['outcome'] = outcome                  
    

# Print the DataFrame to check the results


#stochastic oscillators
all_data['Lowest_5D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 5).min())
all_data['High_5D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 5).max())
all_data['Lowest_15D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 15).min())
all_data['High_15D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 15).max())

all_data['Stochastic_5'] = ((all_data['Close'] - all_data['Lowest_5D'])/(all_data['High_5D'] - all_data['Lowest_5D']))*100
all_data['Stochastic_15'] = ((all_data['Close'] - all_data['Lowest_15D'])/(all_data['High_15D'] - all_data['Lowest_15D']))*100

all_data['Stochastic_%D_5'] = all_data['Stochastic_5'].rolling(window = 5).mean()
all_data['Stochastic_%D_15'] = all_data['Stochastic_5'].rolling(window = 15).mean()

all_data['Stochastic_Ratio'] = all_data['Stochastic_%D_5']/all_data['Stochastic_%D_15']


#RSI
all_data['Diff'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.diff())
all_data['Up'] = all_data['Diff']
all_data.loc[(all_data['Up']<0), 'Up'] = 0

all_data['Down'] = all_data['Diff']
all_data.loc[(all_data['Down']>0), 'Down'] = 0 
all_data['Down'] = abs(all_data['Down'])

all_data['avg_5up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=5).mean())
all_data['avg_5down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=5).mean())

all_data['avg_15up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=15).mean())
all_data['avg_15down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=15).mean())

all_data['RS_5'] = all_data['avg_5up'] / all_data['avg_5down']
all_data['RS_15'] = all_data['avg_15up'] / all_data['avg_15down']

all_data['RSI_5'] = 100 - (100/(1+all_data['RS_5']))
all_data['RSI_15'] = 100 - (100/(1+all_data['RS_15']))

all_data['RSI_ratio'] = all_data['RSI_5']/all_data['RSI_15']


#MACD

all_data['5Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
all_data['15Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=15, adjust=False).mean())
all_data['MACD'] = all_data['15Ewm'] - all_data['5Ewm']

print(all_data)
all_data.to_csv('out.csv')


