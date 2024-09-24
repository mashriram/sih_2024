import numpy as np
import pandas as pd
import plotly.express as px
import torch
from nbeats_pytorch.model import NBeatsNet
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import GRU, LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, header=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index_col=0)

    df.columns = [
        "district",
        "market",
        "commodity",
        "variety",
        "grade",
        "min_price",
        "max_price",
        "modal_price",
        "date",
    ]
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y")
    df = df.sort_values("date")
    return df[["date", "modal_price"]]


df1 = load_and_preprocess_data("response_01_Sep_2014_to_01_Sep_2015.csv")
df2 = load_and_preprocess_data("response_01_Sep_2015_to_01_Sep_2016.csv")

df3 = load_and_preprocess_data("response_01_Sep_2016_to_01_Sep_2017.csv")
df4 = load_and_preprocess_data("response_01_Sep_2017_to_01_Sep_2018.csv")
df5 = load_and_preprocess_data("response_01_Sep_2018_to_01_Sep_2019.csv")

# Combine datasets
df = pd.concat([df1, df2, df3, df4, df5]).sort_values("date")

df = df.groupby(["date"])["modal_price"].agg("mean").reset_index()
# Normalize data
scaler = MinMaxScaler()
df["modal_price_scaled"] = scaler.fit_transform(df[["modal_price"]])


# Prepare data for time series prediction
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : (i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


seq_length = 10
X, y = create_sequences(df["modal_price_scaled"].values, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for LSTM and GRU
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# GRU Model
gru_model = Sequential(
    [Input(shape=(seq_length, 1)), GRU(50, activation="relu"), Dense(1)]
)
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
gru_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, verbose=0)

# LSTM Model
lstm_model = Sequential(
    [Input(shape=(seq_length, 1)), LSTM(50, activation="relu"), Dense(1)]
)
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
lstm_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, verbose=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nbeats_model = NBeatsNet(
    backcast_length=seq_length,
    forecast_length=1,
    stack_types=("generic", "generic"),
    nb_blocks_per_stack=3,
    thetas_dim=(4, 4),
    share_weights_in_stack=False,
    hidden_layer_units=128,
)
nbeats_model.to(device)

optimizer = torch.optim.Adam(nbeats_model.parameters())
criterion = torch.nn.MSELoss()

X_train_tensor = torch.FloatTensor(X_train).to(device)
X_train_tensor = X_train_tensor.view(-1, seq_length, 1)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = nbeats_model(X_train_tensor)

    # Debug information
    print(f"Epoch {epoch}, Output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"Output tuple length: {len(output)}")
        for i, item in enumerate(output):
            print(f"Item {i} shape: {item.shape}")
    else:
        print(f"Output shape: {output.shape}")

    # Handle different output formats
    if isinstance(output, tuple):
        forecast = output[-1]  # Assume the last element is the forecast
    else:
        forecast = output

    loss = criterion(forecast, y_train_tensor)
    loss.backward()
    optimizer.step()

# Make predictions
X_test_tensor = torch.FloatTensor(X_test).to(device)
X_test_tensor = X_test_tensor.view(-1, seq_length, 1).to(device)

with torch.no_grad():
    output = nbeats_model.predict(X_test_tensor)
    print("Inside", output)

# Check if output is a tuple and handle accordingly
if isinstance(output, tuple):
    # Access the forecast based on its expected shape (e.g., first element)
    nbeats_pred = output[0]  # Modify index based on model configuration
else:
    nbeats_pred = output


# Ensure nbeats_pred has the correct shape
if len(nbeats_pred.shape) == 3:
    nbeats_pred = nbeats_pred.squeeze(1)

# Inverse transform predictions

nbeats_pred = scaler.inverse_transform(nbeats_pred)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))


gru_pred = gru_model.predict(X_test_reshaped)
lstm_pred = lstm_model.predict(X_test_reshaped)

X_test_tensor = torch.FloatTensor(X_test).to(device)
X_test_tensor = X_test_tensor.view(-1, seq_length, 1).to(device)

# Inverse transform predictions
gru_pred = scaler.inverse_transform(gru_pred)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))


# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # For F1 score, we need to convert regression to classification
    # Let's consider a prediction correct if it's within 5% of the true value
    y_true_class = (y_true >= np.median(y_true)).astype(int)
    y_pred_class = (y_pred >= np.median(y_pred)).astype(int)
    f1 = f1_score(y_true_class, y_pred_class)

    return mse, mae, rmse, f1, r2


gru_metrics = calculate_metrics(y_test_original, gru_pred)
lstm_metrics = calculate_metrics(y_test_original, lstm_pred)
nbeats_metrics = calculate_metrics(y_test_original, nbeats_pred)

print("GRU Metrics (MSE, MAE, RMSE, F1,R2):", gru_metrics)
print("LSTM Metrics (MSE, MAE, RMSE, F1,R2):", lstm_metrics)
print("N-BEATS Metrics (MSE, MAE, RMSE, F1,R2):", nbeats_metrics)

metrics_df = pd.DataFrame(
    {
        "GRU Metrics": gru_metrics,
        "LSTM Metrics": lstm_metrics,
        "N-BEATS Metrics": nbeats_metrics,
    }
)
metrics_df.rename(index={0: "MSE", 1: "MAE", 2: "RMSE", 3: "F1", 4: "R2"}, inplace=True)
print(metrics_df)
