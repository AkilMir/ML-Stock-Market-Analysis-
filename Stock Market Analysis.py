import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import os

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

# # Create the Random Forest model
# rf_regressor = RandomForestRegressor(
#     n_estimators=100,             # Increased number of trees
#     # bootstrap=True,               # Use bootstrap samples
#     random_state=42               # For reproducibility
# )

# # Set up 10-fold cross-validation
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# # Define custom scorers
# mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
# r2_scorer = make_scorer(r2_score)

# # We need to define the features and target. Let's assume 'Adj Close' is the target
# features_gspc = gspc_df.drop(columns=['Adj Close'])
# target_gspc = gspc_df['Adj Close']

# # Perform cross-validation and compute scores
# mse_scores_gspc = cross_val_score(rf_regressor, features_gspc, target_gspc, cv=kf, scoring=mse_scorer)
# mae_scores_gspc = cross_val_score(rf_regressor, features_gspc, target_gspc, cv=kf, scoring=mae_scorer)
# r2_scores_gspc = cross_val_score(rf_regressor, features_gspc, target_gspc, cv=kf, scoring=r2_scorer)

# # Output results
# print("MSE scores for S&P 500:", -mse_scores_gspc)
# print("MAE scores for S&P 500:", -mae_scores_gspc)
# print("R^2 scores for S&P 500:", r2_scores_gspc)

# # We need to define the features and target. Let's assume 'Adj Close' is the target
# features_ixic = ixic_df.drop(columns=['Adj Close'])
# target_ixic = ixic_df['Adj Close']

# # Perform cross-validation and compute scores
# mse_scores_ixic = cross_val_score(rf_regressor, features_ixic, target_ixic, cv=kf, scoring=mse_scorer)
# mae_scores_ixic = cross_val_score(rf_regressor, features_ixic, target_ixic, cv=kf, scoring=mae_scorer)
# r2_scores_ixic = cross_val_score(rf_regressor, features_ixic, target_ixic, cv=kf, scoring=r2_scorer)

# # Output results
# print("MSE scores for NASDAQ:", -mse_scores_ixic)
# print("MAE scores for NASDAQ:", -mae_scores_ixic)
# print("R^2 scores for NASDAQ:", r2_scores_ixic)

# # We need to define the features and target. Let's assume 'Adj Close' is the target
# features_n225 = n225_df.drop(columns=['Adj Close'])
# target_n225 = n225_df['Adj Close']

# # Perform cross-validation and compute scores
# mse_scores_n225 = cross_val_score(rf_regressor, features_n225, target_n225, cv=kf, scoring=mse_scorer)
# mae_scores_n225 = cross_val_score(rf_regressor, features_n225, target_n225, cv=kf, scoring=mae_scorer)
# r2_scores_n225 = cross_val_score(rf_regressor, features_n225, target_n225, cv=kf, scoring=r2_scorer)

# # Output results
# print("MSE scores for Nikkei 225:", -mse_scores_n225)
# print("MAE scores for Nikkei 225:", -mae_scores_n225)
# print("R^2 scores for Nikkei 225:", r2_scores_n225)

# # Define the MLP regressor

# # Initialize MLPRegressor with specific parameters
# mlp_regressor = MLPRegressor(
#     hidden_layer_sizes=(100, 50),  # Two layers with 100 and 50 neurons
#     activation='tanh',             # 'relu' or 'tanh'
#     solver='adam',                 # 'adam' or 'sgd'
#     alpha=0.001,                   # Regularization strength
#     learning_rate='constant',      # 'constant' or 'adaptive'
#     max_iter=500,                  # Number of iterations
#     random_state=42
# )

# # Applying normalization to the features for the MLP model as it is sensitive to the magnitude of input features
# scaler = StandardScaler()

# def train_and_evaluate_mlp(df, features, target):
#     # Extract features and target from the dataframe
#     X = df[features]
#     y = df[target]
    
#     # Scale the features
#     X_scaled = scaler.fit_transform(X)
    
#     # Perform 10-fold cross-validation
#     mse_scores = cross_val_score(mlp_regressor, X_scaled, y, cv=kf, scoring=mse_scorer)
#     mae_scores = cross_val_score(mlp_regressor, X_scaled, y, cv=kf, scoring=mae_scorer)
#     r2_scores = cross_val_score(mlp_regressor, X_scaled, y, cv=kf, scoring=r2_scorer)
    
#     return -mse_scores, -mae_scores, r2_scores

# # S&P 500 MLP Regression Evaluation
# features_gspc = gspc_df.drop(columns=['Adj Close']).columns.tolist()
# mse_scores_gspc, mae_scores_gspc, r2_scores_gspc = train_and_evaluate_mlp(gspc_df, features_gspc, 'Adj Close')
# print("MLP MSE scores for S&P 500:", mse_scores_gspc)
# print("MLP MAE scores for S&P 500:", mae_scores_gspc)
# print("MLP R^2 scores for S&P 500:", r2_scores_gspc)

# # NASDAQ MLP Regression Evaluation
# features_ixic = ixic_df.drop(columns=['Adj Close']).columns.tolist()
# mse_scores_ixic, mae_scores_ixic, r2_scores_ixic = train_and_evaluate_mlp(ixic_df, features_ixic, 'Adj Close')
# print("MLP MSE scores for NASDAQ:", mse_scores_ixic)
# print("MLP MAE scores for NASDAQ:", mae_scores_ixic)
# print("MLP R^2 scores for NASDAQ:", r2_scores_ixic)

# # Nikkei 225 MLP Regression Evaluation
# features_n225 = n225_df.drop(columns=['Adj Close']).columns.tolist()
# mse_scores_n225, mae_scores_n225, r2_scores_n225 = train_and_evaluate_mlp(n225_df, features_n225, 'Adj Close')
# print("MLP MSE scores for Nikkei 225:", mse_scores_n225)
# print("MLP MAE scores for Nikkei 225:", mae_scores_n225)
# print("MLP R^2 scores for Nikkei 225:", r2_scores_n225)

# # Initialize the Linear Regression model
# lr_regressor = LinearRegression()

# def train_and_evaluate_lr(df, features, target):
#     # Extract features and target from the dataframe
#     X = df[features]
#     y = df[target]
    
#     # Perform 10-fold cross-validation
#     mse_scores = cross_val_score(lr_regressor, X, y, cv=kf, scoring=mse_scorer)
#     mae_scores = cross_val_score(lr_regressor, X, y, cv=kf, scoring=mae_scorer)
#     r2_scores = cross_val_score(lr_regressor, X, y, cv=kf, scoring=r2_scorer)
    
#     return -mse_scores, -mae_scores, r2_scores

# # S&P 500 Linear Regression Evaluation
# features_gspc = gspc_df.drop(columns=['Adj Close']).columns.tolist()
# mse_scores_gspc, mae_scores_gspc, r2_scores_gspc = train_and_evaluate_lr(gspc_df, features_gspc, 'Adj Close')
# print("Linear Regression MSE scores for S&P 500:", mse_scores_gspc)
# print("Linear Regression MAE scores for S&P 500:", mae_scores_gspc)
# print("Linear Regression R^2 scores for S&P 500:", r2_scores_gspc)

# # NASDAQ Linear Regression Evaluation
# features_ixic = ixic_df.drop(columns=['Adj Close']).columns.tolist()
# mse_scores_ixic, mae_scores_ixic, r2_scores_ixic = train_and_evaluate_lr(ixic_df, features_ixic, 'Adj Close')
# print("Linear Regression MSE scores for NASDAQ:", mse_scores_ixic)
# print("Linear Regression MAE scores for NASDAQ:", mae_scores_ixic)
# print("Linear Regression R^2 scores for NASDAQ:", r2_scores_ixic)

# # Nikkei 225 Linear Regression Evaluation
# features_n225 = n225_df.drop(columns=['Adj Close']).columns.tolist()
# mse_scores_n225, mae_scores_n225, r2_scores_n225 = train_and_evaluate_lr(n225_df, features_n225, 'Adj Close')
# print("Linear Regression MSE scores for Nikkei 225:", mse_scores_n225)
# print("Linear Regression MAE scores for Nikkei 225:", mae_scores_n225)
# print("Linear Regression R^2 scores for Nikkei 225:", r2_scores_n225)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

# Perform 10-fold cross-validation for LSTM
def cross_validate_lstm(X, y, n_splits=10, n_steps=30, epochs=20, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores, mae_scores, r2_scores = [], [], []
    
    for train_index, test_index in kf.split(X):
        # Split data into train and test for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the model
        model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluate on the test set
        y_pred = model.predict(X_test).flatten()
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    
    return mse_scores, mae_scores, r2_scores

# Prepare data for each index using only the original features
n_steps = 30  # Number of past days used to predict the next day's price
results = {}

# Iterate through indices and dynamically select available original features
for index_name, df in {"S&P 500": gspc_df, "NASDAQ": ixic_df, "Nikkei 225": n225_df}.items():
    print(f"\nCross-validating for {index_name} with original features only...")
    
    # Dynamically select only existing original features
    available_features = [col for col in ['Open', 'High', 'Low', 'Adj Close', 'Volume'] if col in df.columns]
    original_features_df = df[available_features]
    
    # Create sequences
    X, y = create_lstm_sequences(original_features_df, target_col='Adj Close', n_steps=n_steps)
    
    # Perform 10-fold cross-validation
    mse, mae, r2 = cross_validate_lstm(X, y, n_splits=10, n_steps=n_steps, epochs=20, batch_size=32)
    results[index_name] = {"MSE": mse, "MAE": mae, "R^2": r2}
    
    # Print results
    print(f"{index_name} Results:")
    print(f"Average MSE: {np.mean(mse):.5f}, Average MAE: {np.mean(mae):.5f}, Average R^2: {np.mean(r2):.5f}")

# def create_lstm_sequences(df, target_col, n_steps):
#     X, y = [], []
#     for i in range(len(df) - n_steps):
#         X.append(df.iloc[i:i + n_steps].values)
#         y.append(df.iloc[i + n_steps][target_col])
#     return np.array(X), np.array(y)

# # Define the LSTM model
# def create_lstm_model(input_shape):
#     model = Sequential([
#         LSTM(50, activation='tanh', return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(50, activation='tanh'),
#         Dropout(0.2),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# # Perform 10-fold cross-validation for LSTM
# def cross_validate_lstm(X, y, n_splits=10, n_steps=30, epochs=20, batch_size=32):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     mse_scores, mae_scores, r2_scores = [], [], []
    
#     for train_index, test_index in kf.split(X):
#         # Split data into train and test for this fold
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         # Create and train the model
#         model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
#         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

#         # Evaluate on the test set
#         y_pred = model.predict(X_test).flatten()
#         mse_scores.append(mean_squared_error(y_test, y_pred))
#         mae_scores.append(mean_absolute_error(y_test, y_pred))
#         r2_scores.append(r2_score(y_test, y_pred))
    
#     return mse_scores, mae_scores, r2_scores

# # Prepare data for each index
# n_steps = 30  # Number of past days used to predict the next day's price
# results = {}

# for index_name, df in {"S&P 500": gspc_df, "NASDAQ": ixic_df, "Nikkei 225": n225_df}.items():
#     print(f"\nCross-validating for {index_name}...")
#     X, y = create_lstm_sequences(df, target_col='Adj Close', n_steps=n_steps)
#     mse, mae, r2 = cross_validate_lstm(X, y, n_splits=10, n_steps=n_steps, epochs=20, batch_size=32)
#     results[index_name] = {"MSE": mse, "MAE": mae, "R^2": r2}
#     print(f"{index_name} Results:")
#     print(f"Average MSE: {np.mean(mse):.5f}, Average MAE: {np.mean(mae):.5f}, Average R^2: {np.mean(r2):.5f}")

# Visualize results for each index
# for index_name, metrics in results.items():
#     plt.figure(figsize=(10, 6))
#     plt.plot(metrics['MSE'], label="MSE", marker='o')
#     plt.plot(metrics['MAE'], label="MAE", marker='o')
#     plt.title(f"{index_name} Cross-Validation Metrics")
#     plt.xlabel("Fold")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.show()