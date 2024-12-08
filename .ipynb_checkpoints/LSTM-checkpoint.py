import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# defining tickers and dates 
tickers = ['^GSPC', '^IXIC', '^N225']
start_date = '2010-01-01'
end_date = '2022-01-01'

# getting data from yahoo finance API
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

