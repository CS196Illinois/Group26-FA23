drezer97
drezer97
Online

drezer97 — 09/13/2023 11:39 AM
Hey Matt! Quick question - are you good with having our 30 min weekly sprint meeting on Fridays at 3:30?
realMattyG — 09/13/2023 3:46 PM
yeah that should work
drezer97 — 09/13/2023 3:46 PM
Gang shi no lame shi
Appreciate the response man
realMattyG — 09/13/2023 3:47 PM
denada
realMattyG — 09/14/2023 3:08 PM
@drezer97 hey Andre so I can’t see the group 26 text channel anymore, I only see the voice channel
drezer97 — 09/14/2023 3:35 PM
Hmm are you on your phone or laptop?
realMattyG — 09/14/2023 3:37 PM
youre right yeah i see it now on laptop before i was on my phone. How come it doesn't show up on mobile?
drezer97 — 09/14/2023 3:39 PM
I’m not too sure - I actually also have that problem… I’m going to ask Dhruv about it
Never tried checking the channel via mobile before lol
realMattyG — 09/15/2023 2:55 PM
hey andre, so to add to my project proposal from Monday, we could have the algorithm Optimize times of productivity (tell the algorithm what time of the day you have the most energy, and it will try to generate study times during these periods and not classes, or whether you prefer to study or attend lectures with more energy). The algorithm would also take into account distance between consecutive classes and your mode of transportation (walking, biking, scooter) to make sure you can get to class on time 
other than that, the example you posted pretty much sums up the project pretty well, but the main purpose of the app overall is to help students save a bunch of time registering for classes and tp avoid overcrowded schedules / skipped meals.
realMattyG — 09/22/2023 12:51 PM
@drezer97 so I read the article, and Sophie's and Rahil's responses, and I thought about the questions you posed to us, but I don't really have anything more to add that they didn't cover in their comments. I simply agree that we should focus on a tech-related company since there would be a lot of data out there. Is it fine if I can't come up with more ideas about implementation?
drezer97 — 09/22/2023 12:51 PM
Absolutely! Totally fine
Any questions from the article? I’m gonna cover it a bit during our meeting today
realMattyG — 09/22/2023 12:52 PM
yeah it was a little tough digesting the neural networks but i kind of get the gist and the basic differences between the models
realMattyG — 09/29/2023 10:20 AM
hey andre so i was able to download python and the packages that the article uses, but when I tried running the code it still had some tricky errors that I couldn't figure out. Could you take a look? Here's a screenshot...
Image
realMattyG — 09/29/2023 10:33 AM
also, I'm not gonna lie, I'm having trouble understanding a lot of the code throughout the article...
drezer97 — 09/29/2023 3:16 PM
no worries, let's discuss during or after today's meeting
drezer97 — 10/02/2023 12:18 PM
Hey Matt, could I get your github username?
realMattyG — 10/02/2023 1:00 PM
realMattyG
very original i know
but if it aint broke dont fix it lol
drezer97 — 10/02/2023 5:53 PM
Lmao true true 😤
drezer97 — Yesterday at 8:34 PM
Hey Matt, when you get a chance, could you send me that python file we worked on last week, or upload it to github preferably? It doesn't have to include any work you've put into it since last week's meeting, I just wanted to get Constantine up to speed with what we did so far
realMattyG — Today at 3:32 PM
hey andre this is the file
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader.data as pdr
import seaborn as sns
import matplotlib.pyplot as plt
Expand
test.py
5 KB
drezer97 — Today at 3:39 PM
Awesome thanks Matt!!!
﻿
realMattyG
realMattyG#6147
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
