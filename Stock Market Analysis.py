import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define tickers and date range
tickers = ['^GSPC', '^IXIC', '^N225']
start_date = '2010-01-01'
end_date = '2022-01-01'

# Fetch data
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Drop 'Close' from each ticker in the DataFrame
for ticker in tickers:
    if (ticker, 'Close') in data.columns:  # Check if the 'Close' column exists
        data.drop(columns=[(ticker, 'Close')], inplace=True)  # Drop the 'Close' column

# Apply LOCF for missing values across the entire dataset
data = data.ffill()

# Assuming 'data' is your DataFrame

# Define the window for moving averages and other indicators
window_sma = 30
window_rsi = 14
window_macd_short = 12
window_macd_long = 26
signal_line_window = 9
window_boll = 20

for ticker in tickers:
    # Get the position of 'Adj Close' for each ticker to start inserting new features after it
    position = data.columns.get_loc((ticker, 'Adj Close')) + 1

    # Insert SMA
    sma_series = data[(ticker, 'Adj Close')].rolling(window=window_sma).mean()
    data.insert(loc=position, column=(ticker, 'SMA_30'), value=sma_series)

    # Insert RSI
    delta = data[(ticker, 'Adj Close')].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window_rsi).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window_rsi).mean()
    RS = gain / loss
    rsi_series = 100 - (100 / (1 + RS))
    data.insert(loc=position + 1, column=(ticker, 'RSI_14'), value=rsi_series)

    # Insert MACD and Signal Line
    exp1 = data[(ticker, 'Adj Close')].ewm(span=window_macd_short, adjust=False).mean()
    exp2 = data[(ticker, 'Adj Close')].ewm(span=window_macd_long, adjust=False).mean()
    macd = exp1 - exp2
    data.insert(loc=position + 2, column=(ticker, 'MACD'), value=macd)
    data.insert(loc=position + 3, column=(ticker, 'Signal_Line'), value=macd.ewm(span=signal_line_window, adjust=False).mean())

    # Insert Bollinger Bands
    sma = data[(ticker, 'Adj Close')].rolling(window=window_boll).mean()
    rstd = data[(ticker, 'Adj Close')].rolling(window=window_boll).std()
    data.insert(loc=position + 4, column=(ticker, 'Upper_Band'), value=sma + 2 * rstd)
    data.insert(loc=position + 5, column=(ticker, 'Lower_Band'), value=sma - 2 * rstd)

# Drop rows where any of the specified indicators are NaN
data.dropna(subset=[
    ('^GSPC', 'SMA_30'), 
    ('^GSPC', 'RSI_14'), 
    ('^GSPC', 'Upper_Band'), 
    ('^GSPC', 'Lower_Band'),
    ('^IXIC', 'SMA_30'),
    ('^IXIC', 'RSI_14'),
    ('^IXIC', 'Upper_Band'),
    ('^IXIC', 'Lower_Band'),
    ('^N225', 'SMA_30'),
    ('^N225', 'RSI_14'),
    ('^N225', 'Upper_Band'),
    ('^N225', 'Lower_Band')
    ], inplace=True)

def normalize_data(df, features):
    scaler = MinMaxScaler()
    # Normalize the specified features, ensuring they are referenced correctly
    df.loc[:, features] = scaler.fit_transform(df[features])
    return df

# Apply normalization to each DataFrame, correctly referencing multi-level columns
for ticker in tickers:
    # Specify multi-level columns including the ticker to ensure correct reference
    features_to_normalize = [(ticker, 'Open'), (ticker, 'High'), (ticker, 'Low'), 
                             (ticker, 'Close'), (ticker, 'Adj Close'), (ticker, 'Volume'),
                             (ticker, 'SMA_30'), (ticker, 'Upper_Band'), (ticker, 'Lower_Band')]
    # Filter to ensure only existing columns are selected and passed
    existing_features = [feature for feature in features_to_normalize if feature in data.columns]

    # Check if there are valid features to normalize
    if existing_features:
        # Normalize using the correct multi-level column references
        data.loc[:, existing_features] = normalize_data(data.loc[:, existing_features], existing_features)

# Separate DataFrames for each ticker
gspc_df = data['^GSPC']
ixic_df = data['^IXIC']
n225_df = data['^N225']