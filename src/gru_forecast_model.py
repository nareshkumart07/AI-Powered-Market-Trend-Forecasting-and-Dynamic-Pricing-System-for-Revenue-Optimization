

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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

# Adding Holiday Feature
import holidays

sales_year = daily_sales_df.index.year.unique()[0]
uk_holidays = holidays.UK(years=sales_year)
daily_sales_df['is_holiday'] = daily_sales_df.index.map(lambda x: 1 if x in uk_holidays else 0)

daily_sales_df.head(10)

# Total holiday

total_holidays = daily_sales_df['is_holiday'].sum()
total_holidays

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

# Data Splinting into three part (Train ,Validation and Test)
# Total number of sequences
total_samples = len(X)

# Define split points
train_size = int(total_samples * 0.70)
val_size = int(total_samples * 0.15)
# The rest is the test set

# Split the data
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# converting data into tensor form

# importing libreries

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Convert to PyTorch Tensors (add validation tensors) ---
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)

# Validation tensors
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)

X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)


# Create DataLoaders (add validation loader) ---
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Validation loader
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

"""Difining Model Architecture(Using GRU )"""
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the GRU layer
        # batch_first=True means the input tensor shape is (batch, sequence, feature)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure input is 3D (batch, sequence, feature)
        if x.dim() == 2:
            x = x.unsqueeze(-1) # Add feature dimension if missing

        batch_size = x.size(0)
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # We need to pass the hidden state to the GRU
        out, _ = self.gru(x, h0)

        # We only need the output of the last time step for forecasting
        out = self.fc(out[:, -1, :])
        return out

# --- Model Hyperparameters (using the same as LSTM for comparison) ---
input_size = num_features # Set input_size to the number of features in the scaled data
hidden_size = 128 # Number of neurons in the hidden layer
num_layers = 4 # Number of stacked GRU layers
output_size = 1 # We are predicting a single value

# --- Instantiate the model ---
model = GRUModel(input_size, hidden_size, num_layers, output_size)
print(model)

# FULL TRAINING LOOP WITH EARLY STOPPING

# ---  Initialize variables for Early Stopping ---
# How many epochs to wait after last time validation loss improved.
patience = 20
# A counter for the number of epochs with no improvement
patience_counter = 0
# A variable to store the best validation loss found so far
best_val_loss = float('inf')

# --- Loss Function and Optimizer (no changes here) ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
num_epochs = 100 # We might not run all epochs if we stop early
for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    # --- Check for improvement in validation loss ---
    if avg_val_loss < best_val_loss:
        # If the validation loss improved, update the best loss
        best_val_loss = avg_val_loss
        # Reset the patience counter
        patience_counter = 0
        # --- 3. Save the best model state ---
        # model.state_dict() contains all the learnable parameters (weights and biases)
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ Epoch {epoch+1}: Validation loss improved to {avg_val_loss:.4f}. Saving model.")
    else:
        # If the validation loss did not improve, increment the patience counter
        patience_counter += 1
        print(f"Epoch {epoch+1}: No improvement in validation loss for {patience_counter} epoch(s).")

    # --- 4. Check if we should stop early ---
    if patience_counter >= patience:
        print("!! Early stopping triggered !!")
        break # Exit the training loop

# If avg_val_loss starts increasing while avg_train_loss decreases, your model is overfitting!

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

plt.figure(figsize=(12, 6))
plt.plot(daily_sales_df.index[-len(y_test_actual):], y_test_actual, label='Actual', color='blue')
plt.plot(daily_sales_df.index[-len(y_test_actual):], predictions, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Actual vs. Predicted Quantity')
plt.legend()
