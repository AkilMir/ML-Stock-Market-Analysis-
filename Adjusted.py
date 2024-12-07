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

# Define windows for indicators
window_rsi = 14
window_macd_short = 12
window_macd_long = 26
signal_line_window = 9

# Calculate indicators and keep only less correlated features
for ticker in tickers:
    # Get the position of 'Adj Close' for each ticker to start inserting new features after it
    position = data.columns.get_loc((ticker, 'Adj Close')) + 1

    # Insert RSI
    delta = data[(ticker, 'Adj Close')].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window_rsi).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window_rsi).mean()
    RS = gain / loss
    rsi_series = 100 - (100 / (1 + RS))
    data.insert(loc=position, column=(ticker, 'RSI_14'), value=rsi_series)

    # Insert MACD and Signal Line
    exp1 = data[(ticker, 'Adj Close')].ewm(span=window_macd_short, adjust=False).mean()
    exp2 = data[(ticker, 'Adj Close')].ewm(span=window_macd_long, adjust=False).mean()
    macd = exp1 - exp2
    data.insert(loc=position + 1, column=(ticker, 'MACD'), value=macd)
    data.insert(loc=position + 2, column=(ticker, 'Signal_Line'), value=macd.ewm(span=signal_line_window, adjust=False).mean())

# Drop rows with NaN values for these specific features
data.dropna(subset=[
    ('^GSPC', 'RSI_14'), 
    ('^IXIC', 'RSI_14'), 
    ('^N225', 'RSI_14')
    ], inplace=True)

# Normalize only selected features
def normalize_data(df, features):
    scaler = MinMaxScaler()
    df.loc[:, features] = scaler.fit_transform(df[features])
    return df

# Normalize selected features for each ticker
for ticker in tickers:
    # Specify features to keep
    features_to_normalize = [(ticker, 'Adj Close'), (ticker, 'Volume'),
                             (ticker, 'RSI_14'), (ticker, 'MACD'), (ticker, 'Signal_Line')]
    # Normalize only valid features
    existing_features = [feature for feature in features_to_normalize if feature in data.columns]
    if existing_features:
        data.loc[:, existing_features] = normalize_data(data.loc[:, existing_features], existing_features)

# Separate DataFrames for each ticker
gspc_df = data['^GSPC']
ixic_df = data['^IXIC']
n225_df = data['^N225']

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Create sequences for LSTM
def create_lstm_sequences(df, target_col, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:i + n_steps].values)
        y.append(df.iloc[i + n_steps][target_col])
    return np.array(X), np.array(y)

# Define the LSTM model
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

# Set parameters
n_steps = 30  # Number of past days used to predict the next day's price
epochs = 20
batch_size = 32

# Prepare data for each index
results = {}
for index_name, df in {"S&P 500": gspc_df, "NASDAQ": ixic_df, "Nikkei 225": n225_df}.items():
    print(f"\nTraining and evaluating for {index_name} with raw features...")

    # Select only raw features
    selected_features = ['Adj Close', 'Volume']
    available_features = [col for col in selected_features if col in df.columns]
    selected_features_df = df[available_features]

    # Add random noise to simulate real-world unpredictability
    selected_features_df['Adj Close'] += np.random.normal(0, 0.01, len(selected_features_df))

    # Split into train (2010–2017) and test (2018–2022)
    train_data = selected_features_df[selected_features_df.index < '2018-01-01']
    test_data = selected_features_df[selected_features_df.index >= '2018-01-01']

    # Create LSTM sequences
    X_train, y_train = create_lstm_sequences(train_data, target_col='Adj Close', n_steps=n_steps)
    X_test, y_test = create_lstm_sequences(test_data, target_col='Adj Close', n_steps=n_steps)

    # Train the model
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[index_name] = {"MSE": mse, "MAE": mae, "R^2": r2}

    # Print results
    print(f"{index_name} Results:")
    print(f"Test MSE: {mse:.5f}, Test MAE: {mae:.5f}, Test R^2: {r2:.5f}")

# Display all results
print("\nFinal Results for All Indices:")
for index_name, metrics in results.items():
    print(f"{index_name} - MSE: {metrics['MSE']:.5f}, MAE: {metrics['MAE']:.5f}, R^2: {metrics['R^2']:.5f}")

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Prepare data for each index
results_svm = {}
results_xgb = {}
n_steps = 30  # Number of past days used to predict the next day's price

# Create sequences for SVM and XGBoost (requires reshaping)
def create_sequences_sklearn(df, target_col, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df.iloc[i:i + n_steps].values.flatten())  # Flatten time steps into 1D for sklearn
        y.append(df.iloc[i + n_steps][target_col])
    return np.array(X), np.array(y)

# Standardize the features for SVM and XGBoost
scaler = StandardScaler()

for index_name, df in {"S&P 500": gspc_df, "NASDAQ": ixic_df, "Nikkei 225": n225_df}.items():
    print(f"\nTraining and evaluating SVM and XGBoost for {index_name} with raw features...")

    # Select only raw features
    selected_features = ['Adj Close', 'Volume']
    available_features = [col for col in selected_features if col in df.columns]
    selected_features_df = df[available_features]

    # Split into train (2010–2017) and test (2018–2022)
    train_data = selected_features_df[selected_features_df.index < '2018-01-01']
    test_data = selected_features_df[selected_features_df.index >= '2018-01-01']

    # Create sequences
    X_train, y_train = create_sequences_sklearn(train_data, target_col='Adj Close', n_steps=n_steps)
    X_test, y_test = create_sequences_sklearn(test_data, target_col='Adj Close', n_steps=n_steps)

    # Standardize the features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------- Train SVM Model ----------
    print(f"Training SVM for {index_name}...")
    svm_model = SVR(kernel='rbf', C=100, epsilon=0.01)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    mse_svm = mean_squared_error(y_test, y_pred_svm)
    mae_svm = mean_absolute_error(y_test, y_pred_svm)
    r2_svm = r2_score(y_test, y_pred_svm)
    results_svm[index_name] = {"MSE": mse_svm, "MAE": mae_svm, "R^2": r2_svm}
    print(f"SVM {index_name} Results: MSE: {mse_svm:.5f}, MAE: {mae_svm:.5f}, R^2: {r2_svm:.5f}")

    # ---------- Train XGBoost Model ----------
    print(f"Training XGBoost for {index_name}...")
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    results_xgb[index_name] = {"MSE": mse_xgb, "MAE": mae_xgb, "R^2": r2_xgb}
    print(f"XGBoost {index_name} Results: MSE: {mse_xgb:.5f}, MAE: {mae_xgb:.5f}, R^2: {r2_xgb:.5f}")

# Display Final Results
print("\nFinal Results for SVM:")
for index_name, metrics in results_svm.items():
    print(f"{index_name} - MSE: {metrics['MSE']:.5f}, MAE: {metrics['MAE']:.5f}, R^2: {metrics['R^2']:.5f}")

print("\nFinal Results for XGBoost:")
for index_name, metrics in results_xgb.items():
    print(f"{index_name} - MSE: {metrics['MSE']:.5f}, MAE: {metrics['MAE']:.5f}, R^2: {metrics['R^2']:.5f}")



