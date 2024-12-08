import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Configuration
# -----------------------------
tickers = ['^GSPC', '^IXIC', '^N225']
start_date = '2010-01-01'
end_date = '2022-01-01'
train_end_date = '2018-01-01'  # Split date

n_steps = 30
epochs = 20
batch_size = 32

selected_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI_14', 'MACD', 'Signal_Line']

# -----------------------------
# Data Fetching
# -----------------------------
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

# Remove 'Close' columns to avoid confusion (already have Adj Close)
for ticker in tickers:
    if (ticker, 'Close') in data.columns:
        data.drop(columns=[(ticker, 'Close')], inplace=True)

# Forward-fill missing values
data = data.ffill()

# -----------------------------
# Compute Technical Indicators
# -----------------------------
window_rsi = 14
window_macd_short = 12
window_macd_long = 26
signal_line_window = 9

for ticker in tickers:
    # Compute RSI
    delta = data[(ticker, 'Adj Close')].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window_rsi).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window_rsi).mean()
    RS = gain / loss
    rsi_series = 100 - (100 / (1 + RS))

    # Insert RSI after Adj Close column
    adj_close_col_pos = data.columns.get_loc((ticker, 'Adj Close')) + 1
    data.insert(loc=adj_close_col_pos, column=(ticker, 'RSI_14'), value=rsi_series)

    # Compute MACD
    exp1 = data[(ticker, 'Adj Close')].ewm(span=window_macd_short, adjust=False).mean()
    exp2 = data[(ticker, 'Adj Close')].ewm(span=window_macd_long, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal_line_window, adjust=False).mean()

    # Insert MACD and Signal Line after RSI column
    rsi_col_pos = data.columns.get_loc((ticker, 'RSI_14')) + 1
    data.insert(loc=rsi_col_pos, column=(ticker, 'MACD'), value=macd)
    data.insert(loc=rsi_col_pos + 1, column=(ticker, 'Signal_Line'), value=signal_line)

# Drop rows with NaN (due to RSI/MACD calculations)
data.dropna(inplace=True)

# -----------------------------
# Train/Test Split Before Scaling
# -----------------------------
train_data = data[data.index < train_end_date]
test_data = data[data.index >= train_end_date]

# We'll scale each ticker's features separately for clarity.
# Store scalers so we can apply the same scaler to test data.
scalers = {}

def scale_ticker_data(train_df, test_df, ticker):
    # Extract the features for the current ticker
    ticker_features = [(ticker, f) for f in selected_features if (ticker, f) in train_df.columns]

    if not ticker_features:
        return train_df, test_df  # In case some data is missing

    # Fit scaler on training data only
    scaler = MinMaxScaler()
    train_values = train_df.loc[:, ticker_features].values
    train_scaled = scaler.fit_transform(train_values)

    # Apply to test data
    test_values = test_df.loc[:, ticker_features].values
    test_scaled = scaler.transform(test_values)

    # Replace original columns with scaled values
    train_df.loc[:, ticker_features] = train_scaled
    test_df.loc[:, ticker_features] = test_scaled

    scalers[ticker] = scaler
    return train_df, test_df

# Scale data ticker by ticker
for ticker in tickers:
    train_data, test_data = scale_ticker_data(train_data, test_data, ticker)

# Separate DataFrames for each ticker after scaling
gspc_df = pd.concat([train_data['^GSPC'], test_data['^GSPC']])
ixic_df = pd.concat([train_data['^IXIC'], test_data['^IXIC']])
n225_df = pd.concat([train_data['^N225'], test_data['^N225']])

# -----------------------------
# Sequence Creation Functions
# -----------------------------
def create_lstm_sequences(df, target_col, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:i+n_steps][selected_features].values)
        y.append(df.iloc[i+n_steps][target_col])
    return np.array(X), np.array(y)

def create_sequences_sklearn(df, target_col, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        seq_features = df.iloc[i:i+n_steps][selected_features].values.flatten()
        X.append(seq_features)
        y.append(df.iloc[i+n_steps][target_col])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='tanh'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# -----------------------------
# Training and Evaluation
# -----------------------------
results = {"LSTM": {}, "SVM": {}, "XGBoost": {}}

indices_dict = {
    "S&P 500": gspc_df,
    "NASDAQ": ixic_df,
    "Nikkei 225": n225_df
}

for index_name, df in indices_dict.items():
    print(f"\nTraining and evaluating models for {index_name}...")

    # Split back into train/test using the chosen date
    train_idx = df.index < train_end_date
    test_idx = df.index >= train_end_date
    train_subset = df[train_idx].copy()
    test_subset = df[test_idx].copy()

    # LSTM
    X_train_lstm, y_train_lstm = create_lstm_sequences(train_subset, target_col='Adj Close', n_steps=n_steps)
    X_test_lstm, y_test_lstm = create_lstm_sequences(test_subset, target_col='Adj Close', n_steps=n_steps)
    if X_train_lstm.size == 0 or X_test_lstm.size == 0:
        print(f"Not enough data for LSTM for {index_name}, skipping.")
        continue
    lstm_model = create_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]))
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, verbose=1)
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
    results["LSTM"][index_name] = {
        "MSE": mean_squared_error(y_test_lstm, y_pred_lstm),
        "MAE": mean_absolute_error(y_test_lstm, y_pred_lstm),
        "R^2": r2_score(y_test_lstm, y_pred_lstm),
    }

    # SVM
    X_train_svm, y_train_svm = create_sequences_sklearn(train_subset, target_col='Adj Close', n_steps=n_steps)
    X_test_svm, y_test_svm = create_sequences_sklearn(test_subset, target_col='Adj Close', n_steps=n_steps)
    if X_train_svm.size == 0 or X_test_svm.size == 0:
        print(f"Not enough data for SVM for {index_name}, skipping.")
        continue

    # No double scaling here; data already scaled
    svm_model = SVR(kernel='rbf', C=100, epsilon=0.01)
    svm_model.fit(X_train_svm, y_train_svm)
    y_pred_svm = svm_model.predict(X_test_svm)
    results["SVM"][index_name] = {
        "MSE": mean_squared_error(y_test_svm, y_pred_svm),
        "MAE": mean_absolute_error(y_test_svm, y_pred_svm),
        "R^2": r2_score(y_test_svm, y_pred_svm),
    }

    # XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xgb_model.fit(X_train_svm, y_train_svm)
    y_pred_xgb = xgb_model.predict(X_test_svm)
    results["XGBoost"][index_name] = {
        "MSE": mean_squared_error(y_test_svm, y_pred_xgb),
        "MAE": mean_absolute_error(y_test_svm, y_pred_xgb),
        "R^2": r2_score(y_test_svm, y_pred_xgb),
    }

# -----------------------------
# Print Final Results
# -----------------------------
for model_name, model_results in results.items():
    print(f"\nFinal Results for {model_name}:")
    for index_name, metrics in model_results.items():
        print(f"{index_name} - MSE: {metrics['MSE']:.5f}, MAE: {metrics['MAE']:.5f}, R^2: {metrics['R^2']:.5f}")



