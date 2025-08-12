
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/refine_file.csv")
df.head()

# Ensure 'InvoiceDate' is a datetime object
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')

# Set InvoiceDate as the index to make time-based operations easier
df.set_index('InvoiceDate', inplace=True)

# Let's focus on 20 top-selling product
top_products = df['StockCode'].value_counts().head(20)
top_products.head(5)

# (As we identified that 'WHITE HANGING HEART T-LIGHT HOLDER' is a top product)
product_df = df[df['StockCode'] == '85099B'].copy()

# Aggregate daily
daily_sales_df = product_df.resample('D')['Quantity'].sum().to_frame()

# FEATURE ENGINEERING
daily_sales_df['day_of_week'] = daily_sales_df.index.dayofweek
daily_sales_df['month'] = daily_sales_df.index.month
daily_sales_df['lag_1_day'] = daily_sales_df['Quantity'].shift(1)
daily_sales_df['lag_7_days'] = daily_sales_df['Quantity'].shift(7)
daily_sales_df['lag_30_days'] = daily_sales_df['Quantity'].shift(30)
daily_sales_df['rolling_mean_7_days'] = daily_sales_df['Quantity'].shift(1).rolling(window=7).mean()
daily_sales_df['rolling_mean_30_days'] = daily_sales_df['Quantity'].shift(1).rolling(window=30).mean()
daily_sales_df['rolling_std_7_days'] = daily_sales_df['Quantity'].shift(1).rolling(window=7).std()
daily_sales_df['rolling_std_30_days'] = daily_sales_df['Quantity'].shift(1).rolling(window=30).std()

# Fill NaNs from lag/rolling
daily_sales_df.fillna(0, inplace=True)

# SCALING

from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(daily_sales_df)
scaled_features = np.array(scaled_features, dtype=np.float32)

num_features = scaled_features.shape[1]
target_col_idx = daily_sales_df.columns.get_loc("Quantity")

# Define the forecast horizon
forecast_horizon = 1  # Number of days ahead to forecast
seq_length = 60      # Number of past days to use as input

X ,y = [],[]

for i in range(len(scaled_features) - seq_length - forecast_horizon):
  X.append(scaled_features[i:i+seq_length])
  y.append(scaled_features[i+seq_length:i+seq_length+forecast_horizon, target_col_idx])

X,y = np.array(X), np.array(y)

X.shape , y.shape

# Define a split point (e.g., 80% for training)
training_size = int(len(X) * 0.8)

X_train, X_test = X[:training_size], X[training_size:]
y_train, y_test = y[:training_size], y[training_size:]

# converting data into tensor form

# importing libreries

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Convert to PyTorch Tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1) # Reshape y
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1) # Reshape y

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # No shuffle for time series
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""Difining Model Architecture(Using LSTM )"""

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer
        # batch_first=True means the input tensor shape is (batch, sequence, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure input is 3D (batch, sequence, feature)
        if x.dim() == 2:
            x = x.unsqueeze(-1) # Add feature dimension if missing

        batch_size = x.size(0)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # We need to pass the hidden and cell states to the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # We only need the output of the last time step for forecasting
        out = self.fc(out[:, -1, :])
        return out

# --- Model Hyperparameters ---
input_size = num_features # Set input_size to the number of features in the scaled data
hidden_size = 60 # Number of neurons in the hidden layer
num_layers = 3 # Number of stacked LSTM layers
output_size = 1 # We are predicting a single value

# --- Instantiate the model ---
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
print(model)

# --- Loss Function and Optimizer ---
criterion = nn.MSELoss() # Mean Squared Error is good for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
num_epochs = 100
for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 2. Backward and optimize
        optimizer.zero_grad() # Clear gradients from previous iteration
        loss.backward() # Compute gradients
        optimizer.step() # Update weights

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Set the model to evaluation mode
model.eval()
all_predictions = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        outputs = model(batch_X)
        all_predictions.append(outputs.numpy())

# Concatenate predictions from all batches
predictions_scaled = np.concatenate(all_predictions)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Undo the scaling
# Create dummy arrays with the correct number of features (7)
predictions_full = np.zeros((predictions_scaled.shape[0], num_features))
y_test_actual_full = np.zeros((y_test_tensor.numpy().shape[0], num_features))

# Place the scaled predictions and actuals into the target column
predictions_full[:, target_col_idx] = predictions_scaled[:, 0]
y_test_actual_full[:, target_col_idx] = y_test_tensor.numpy()[:, 0]

# Inverse transform the full arrays
predictions = scaler.inverse_transform(predictions_full)[:, target_col_idx]
y_test_actual = scaler.inverse_transform(y_test_actual_full)[:, target_col_idx]


# Calculate error metrics
mae = mean_absolute_error(y_test_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'Test MAE: {mae:.2f}')
print(f'Test RMSE: {rmse:.2f}')

# Visualize the results

plt.figure(figsize=(15, 6))
plt.plot(y_test_actual, color='blue', label='Actual Sales')
plt.plot(predictions, color='red', linestyle='--', label='Forecasted Sales')
plt.title('PyTorch LSTM Forecast vs. Actuals')
plt.xlabel('Time')
plt.ylabel('Quantity Sold')
plt.legend()
plt.show()
