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

rawData = pd.read_csv("GOOG_full_1hour_adjsplitdiv.csv", names = ["Date", "Open", "High", "Low", "Close", "Volume"])
