import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

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

# Create the Random Forest model
rf_regressor = RandomForestRegressor(
    n_estimators=100,             # Increased number of trees
    # bootstrap=True,               # Use bootstrap samples
    random_state=42               # For reproducibility
)

# Set up 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Define custom scorers
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# We need to define the features and target. Let's assume 'Adj Close' is the target
features_gspc = gspc_df.drop(columns=['Adj Close'])
target_gspc = gspc_df['Adj Close']

# Perform cross-validation and compute scores
mse_scores_gspc = cross_val_score(rf_regressor, features_gspc, target_gspc, cv=kf, scoring=mse_scorer)
mae_scores_gspc = cross_val_score(rf_regressor, features_gspc, target_gspc, cv=kf, scoring=mae_scorer)
r2_scores_gspc = cross_val_score(rf_regressor, features_gspc, target_gspc, cv=kf, scoring=r2_scorer)

# Output results
print("MSE scores for S&P 500:", -mse_scores_gspc)
print("MAE scores for S&P 500:", -mae_scores_gspc)
print("R^2 scores for S&P 500:", r2_scores_gspc)

# We need to define the features and target. Let's assume 'Adj Close' is the target
features_ixic = ixic_df.drop(columns=['Adj Close'])
target_ixic = ixic_df['Adj Close']

# Perform cross-validation and compute scores
mse_scores_ixic = cross_val_score(rf_regressor, features_ixic, target_ixic, cv=kf, scoring=mse_scorer)
mae_scores_ixic = cross_val_score(rf_regressor, features_ixic, target_ixic, cv=kf, scoring=mae_scorer)
r2_scores_ixic = cross_val_score(rf_regressor, features_ixic, target_ixic, cv=kf, scoring=r2_scorer)

# Output results
print("MSE scores for NASDAQ:", -mse_scores_ixic)
print("MAE scores for NASDAQ:", -mae_scores_ixic)
print("R^2 scores for NASDAQ:", r2_scores_ixic)

# We need to define the features and target. Let's assume 'Adj Close' is the target
features_n225 = n225_df.drop(columns=['Adj Close'])
target_n225 = n225_df['Adj Close']

# Perform cross-validation and compute scores
mse_scores_n225 = cross_val_score(rf_regressor, features_n225, target_n225, cv=kf, scoring=mse_scorer)
mae_scores_n225 = cross_val_score(rf_regressor, features_n225, target_n225, cv=kf, scoring=mae_scorer)
r2_scores_n225 = cross_val_score(rf_regressor, features_n225, target_n225, cv=kf, scoring=r2_scorer)

# Output results
print("MSE scores for Nikkei 225:", -mse_scores_n225)
print("MAE scores for Nikkei 225:", -mae_scores_n225)
print("R^2 scores for Nikkei 225:", r2_scores_n225)