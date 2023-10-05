import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as pdr
import seaborn as sns
import matplotlib.pyplot as plt
import bs4 as bs
import requests
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

#extracting data from Yahoo Finance API
#ticker = company identifier
tickers = ['AAPL','NFLX']

#Data Frame = basically 2d array with column and row identifiers

all_data = pd.DataFrame()
test_data = pd.DataFrame()
no_data = []

yf.pdr_override()

for i in tickers:
    #make API call using get_data_yahoo
    
    test_data = pdr.get_data_yahoo(i, start = dt.datetime(2019,1,1), end = dt.datetime(2019,1,30))
    #adding another column to differentiate the tickers
    
    test_data['symbol'] = i
    all_data = pd.concat([all_data, test_data])


#Creating Return column
all_data['return'] = all_data.groupby('symbol')['Close'].pct_change()




#takes continual 5-day averages and 15-day averages throughout the whole "all_data" dataframe

all_data['SMA_5'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 5).mean())
all_data['SMA_15'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 15).mean())

#takes the ratio of the 15-day average to the 5-day average (if ratio < 1, that means that the stock is above average recently (you should sell soon since you already have a profit)

all_data['SMA_ratio'] = all_data['SMA_15'] / all_data['SMA_5']

#creates excell spreadsheet representation of the data
all_data.to_csv('all_data.csv')



#Plotting
#just copy from matplotlib and change parameters as necessary

start = dt.datetime.strptime('2019-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2019-12-31', '%Y-%m-%d')
sns.set()

fig = plt.figure(facecolor = 'white', figsize = (20,10))

ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
ax0.plot(all_data[all_data.symbol=='AAPL'].loc[start:end,['Close','SMA_5','SMA_15']])
ax0.set_facecolor('ghostwhite')
ax0.legend(['Close','SMA_5','SMA_15'],ncol=3, loc = 'upper left', fontsize = 15)
plt.title("Apple Stock Price, Slow and Fast Moving Average", fontsize = 20)

ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
ax1.plot(all_data[all_data.symbol=='AAPL'].loc[start:end,['SMA_ratio']], color = 'blue')
ax1.legend(['SMA_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
ax1.set_facecolor('silver')
plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
plt.show()
'''
sns.set()



#Obtain list of S&100 companies from wikipedia
resp = requests.get("https://en.wikipedia.org/wiki/S%26P_100")
convert_soup = bs.BeautifulSoup(resp.text, 'lxml')
table = convert_soup.find('table',{'class':'wikitable sortable'})

tickers = []

for rows in table.findAll('tr')[1:]:
    ticker = rows.findAll('td')[0].text.strip()
    tickers.append(ticker)

all_data = pd.DataFrame()
test_data = pd.DataFrame()
no_data = []

#Extract data from Yahoo Finance
for i in tickers:
    try:
        print(i)
        test_data = pdr.get_data_yahoo(i, start = dt.datetime(1990,1,1), end = dt.date.today())
        test_data['symbol'] = i
        all_data = all_data.append(test_data)
        clear_output(wait = True)
    except:
        no_data.append(i)

    clear_output(wait = True)


Target_variables = ['SMA_ratio','ATR_5','ATR_15','ATR_Ratio',
                       'ADX_5','ADX_15','SMA_Volume_Ratio','Stochastic_5','Stochastic_15','Stochastic_Ratio',
                      'RSI_5','RSI_15','RSI_ratio','MACD']
for variable in Target_variables:
    all_data.loc[:,variable] = mstats.winsorize(all_data.loc[:,variable], limits = [0.1,0.1])
'''
test.py
5 KB
