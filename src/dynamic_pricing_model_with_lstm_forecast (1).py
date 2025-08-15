
"""
Forecast + Dynamic Pricing (LSTM + Elasticity)
----------------------------------------------
- Baseline demand forecast: LSTM (1-day ahead) with engineered features.
- Price sensitivity: log-linear elasticity estimated from history.
- Dynamic price: maximize p * demand(p) using the elasticity-adjusted forecast.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
import itertools

# ==============================================================================
# Part 1: Demand Forecasting (Based on your provided code)
# ==============================================================================
# This section is largely the same as the script you provided. It handles
# data loading, feature engineering, model training, and sales forecasting.
# I've added comments and organized it for clarity.
# ==============================================================================

# --- 1. Setup and Reproducibility ---
print("--- Part 1: Demand Forecasting ---")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- 2. Data Loading and Initial Preparation ---
try:
    # Attempt to load the dataset
    # IMPORTANT: Please replace "/content/drive/MyDrive/refine_file.csv" with the actual path to your CSV file.
    df = pd.read_csv("/content/drive/MyDrive/refine_file.csv")
except FileNotFoundError:
    print("\nERROR: 'refine_file.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script, or provide the full path.\n")
    # Create a dummy dataframe to prevent the script from crashing immediately
    df = pd.DataFrame({
        'InvoiceDate': pd.to_datetime(['2011-12-01', '2011-12-02']),
        'StockCode': ['85099B', '85099B'],
        'Quantity': [10, 15],
        'UnitPrice': [1.95, 1.95]
    })
    print("Continuing with a dummy dataframe for demonstration purposes.\n")


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df.set_index('InvoiceDate', inplace=True)

# --- 3. Isolate a Top-Selling Product ---
# We focus on one product to forecast its demand.
# Your original code identified '85099B' as a top product.
PRODUCT_ID = '85099B'
product_df = df[df['StockCode'] == PRODUCT_ID].copy()

# Aggregate sales data to a daily level
daily_sales_df = product_df.resample('D')['Quantity'].sum().to_frame()

# --- 4. Feature Engineering ---
# We create features that might help the model predict future sales.
print("Creating time-series features...")
daily_sales_df['day_of_week'] = daily_sales_df.index.dayofweek
daily_sales_df['month'] = daily_sales_df.index.month
daily_sales_df['lag_1_day'] = daily_sales_df['Quantity'].shift(1)
daily_sales_df['lag_7_days'] = daily_sales_df['Quantity'].shift(7)
daily_sales_df['rolling_mean_7_days'] = daily_sales_df['Quantity'].shift(1).rolling(window=7).mean()
daily_sales_df['rolling_std_7_days'] = daily_sales_df['Quantity'].shift(1).rolling(window=7).std()

# Add a holiday feature for the UK
sales_years = daily_sales_df.index.year.unique()
uk_holidays = holidays.UK(years=sales_years)
daily_sales_df['is_holiday'] = daily_sales_df.index.map(lambda x: 1 if x in uk_holidays else 0)

# --- NEW: Load and Merge Competitor Price Data ---
try:
    competitor_df = pd.read_csv("competitor_prices.csv")
    product_competitor_prices = competitor_df[competitor_df['StockCode'] == PRODUCT_ID]

    if not product_competitor_prices.empty:
        # Add competitor prices as new features.
        # Since we have one price per competitor for the product, we assign it as a constant column.
        daily_sales_df['competitor_a_price'] = product_competitor_prices['Competitor_A_Price'].iloc[0]
        daily_sales_df['competitor_b_price'] = product_competitor_prices['Competitor_B_Price'].iloc[0]
        daily_sales_df['competitor_c_price'] = product_competitor_prices['Competitor_C_Price'].iloc[0]
        print("Successfully merged competitor price data as new features.")
    else:
        print(f"Warning: Product ID {PRODUCT_ID} not found in competitor_prices.csv. Using fallback values.")
        daily_sales_df['competitor_a_price'] = 1.90
        daily_sales_df['competitor_b_price'] = 2.00
        daily_sales_df['competitor_c_price'] = 2.10
except FileNotFoundError:
    print("Warning: 'competitor_prices.csv' not found. Using fallback values for competitor features.")
    daily_sales_df['competitor_a_price'] = 1.90
    daily_sales_df['competitor_b_price'] = 2.00
    daily_sales_df['competitor_c_price'] = 2.10

# Fill any missing values that resulted from lags and rolling windows
daily_sales_df.fillna(0, inplace=True)

# --- 5. Data Scaling and Sequencing ---
print("Scaling features and creating sequences for the LSTM model...")
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(daily_sales_df)

# Prepare data for the LSTM model
seq_length = 60  # Use 60 days of history to predict the next day
X, y = [], []
target_col_idx = daily_sales_df.columns.get_loc("Quantity")

for i in range(len(scaled_features) - seq_length):
    X.append(scaled_features[i:i + seq_length])
    y.append(scaled_features[i + seq_length, target_col_idx])

X, y = np.array(X), np.array(y)

# --- 6. Data Splitting (Train, Validation, Test) ---
total_samples = len(X)
train_size = int(total_samples * 0.70)
val_size = int(total_samples * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# --- 7. PyTorch DataLoaders ---
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_val_tensor = torch.from_numpy(X_val).float()
y_val_tensor = torch.from_numpy(y_val).float().view(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

# --- 8. LSTM Model Architecture ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # The LSTM returns the output and the final hidden/cell states
        out, _ = self.lstm(x)
        # We pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# --- 8. Grid Search for Hyperparameter Tuning ---
# ==============================================================================
print("\n--- Starting Grid Search for Best Hyperparameters ---")

# Define the grid of parameters to search
param_grid = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2],
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64]
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

best_params = {}
best_global_val_loss = float('inf')
input_size = X_train.shape[2]

# Loop through each combination of parameters
for params in all_params:
    print(f"\nTesting parameters: {params}")

    # Create DataLoaders with the current batch size
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    # Instantiate the model with current parameters
    model = LSTMModel(input_size=input_size, hidden_size=params['hidden_size'], num_layers=params['num_layers'], output_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop with early stopping for this parameter set
    patience = 10
    num_epochs = 50 # Reduced epochs for faster grid search, can be increased
    best_local_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_local_val_loss:
            best_local_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} for this set.")
            break

    # Check if this parameter set is the best one so far
    if best_local_val_loss < best_global_val_loss:
        best_global_val_loss = best_local_val_loss
        best_params = params
        torch.save(model.state_dict(), 'best_forecasting_model.pth')
        print(f"✅ New best model found with validation loss: {best_global_val_loss:.4f}")

print(f"\n--- Grid Search Complete ---")
print(f"Best parameters found: {best_params}")
print(f"Best validation loss: {best_global_val_loss:.4f}")

# --- 9. Final Model Evaluation ---
# Load the best model found during the grid search
best_model = LSTMModel(input_size=input_size, hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], output_size=1)
best_model.load_state_dict(torch.load('best_forecasting_model.pth'))
best_model.eval()

# Create a test loader with the best batch size
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=best_params['batch_size'], shuffle=False)

# --- 10. Model Evaluation ---
predictions_scaled = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        outputs = best_model(batch_X)
        predictions_scaled.extend(outputs.numpy())

predictions_scaled = np.array(predictions_scaled).flatten()

# Inverse-transform the predictions to get the actual sales quantity
dummy_array = np.zeros((len(predictions_scaled), scaled_features.shape[1]))
dummy_array[:, target_col_idx] = predictions_scaled
forecasted_demand = scaler.inverse_transform(dummy_array)[:, target_col_idx]

# Inverse-transform the actual test values for comparison
dummy_array_actual = np.zeros((len(y_test), scaled_features.shape[1]))
dummy_array_actual[:, target_col_idx] = y_test
actual_demand = scaler.inverse_transform(dummy_array_actual)[:, target_col_idx]

# Calculate error metrics
mae = mean_absolute_error(actual_demand, forecasted_demand)
rmse = np.sqrt(mean_squared_error(actual_demand, forecasted_demand))
print(f'\nOptimized Forecast Model Evaluation:')
print(f'Test Mean Absolute Error (MAE): {mae:.2f}')
print(f'Test Root Mean Squared Error (RMSE): {rmse:.2f}')

"""Part 2: Dynamic Pricing Model"""

# Now, we use the demand forecast from Part 1 to suggest optimal prices.
# The core idea is simple:
# - If we predict HIGH demand, we can slightly INCREASE the price.
# - If we predict LOW demand, we should slightly DECREASE the price to
#   encourage more sales.

print("\n--- Part 2: Dynamic Pricing Logic ---")

def calculate_dynamic_price(base_price, forecasted_demand, baseline_demand, elasticity=-1.5, intensity=0.1):
    demand_diff = (forecasted_demand - baseline_demand) / baseline_demand
    price_change_factor = (demand_diff / elasticity) * intensity
    new_price = base_price * (1 + price_change_factor)
    max_price = base_price * 1.25
    min_price = base_price * 0.75
    return max(min(new_price, max_price), min_price)

# --- 1. Define Baseline Metrics ---
if not product_df['Price'].empty and product_df['Price'].mean() > 0:
    BASE_PRICE = product_df['Price'].mean()
else:
    BASE_PRICE = 1.95 # Fallback value

if not daily_sales_df['Quantity'].empty:
    BASELINE_DEMAND = daily_sales_df['Quantity'].mean()
else:
    BASELINE_DEMAND = 12 # Fallback value

print(f"Established Baselines for Product '{PRODUCT_ID}':")
print(f"  - Base Price: ${BASE_PRICE:.2f}")
print(f"  - Baseline Average Daily Demand: {BASELINE_DEMAND:.2f} units")

# --- 2. Generate Dynamic Prices ---
# Now, we loop through our forecasted demand and calculate a suggested price for each day.
suggested_prices = []
for demand_forecast in forecasted_demand:
    new_price = calculate_dynamic_price(
        base_price=BASE_PRICE,
        forecasted_demand=demand_forecast,
        baseline_demand=BASELINE_DEMAND
    )
    suggested_prices.append(new_price)

# --- 3. Visualize the Results ---
print("\nVisualizing the results...")

# Get the dates corresponding to the test set
test_dates = daily_sales_df.index[-len(actual_demand):]

fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot 1: Demand (Actual vs. Forecasted)
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Quantity Sold', color=color)
ax1.plot(test_dates, actual_demand, label='Actual Demand', color=color, linestyle='--', alpha=0.7)
ax1.plot(test_dates, forecasted_demand, label='Forecasted Demand', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')
ax1.set_title(f'Forecasted Demand & Dynamic Price for Product: {PRODUCT_ID}')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Suggested Price ($)', color=color)
ax2.plot(test_dates, suggested_prices, label='Suggested Dynamic Price', color=color, linewidth=2, marker='o', markersize=4, linestyle='-')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=BASE_PRICE, color='red', linestyle=':', label=f'Base Price (${BASE_PRICE:.2f})')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.show()

# --- 4. Display a Sample of the Results ---
results_df = pd.DataFrame({
    'Date': test_dates,
    'Actual_Demand': actual_demand,
    'Forecasted_Demand': forecasted_demand,
    'Suggested_Price': suggested_prices
})
results_df['Forecasted_Demand'] = results_df['Forecasted_Demand'].round(0)
results_df['Suggested_Price'] = results_df['Suggested_Price'].round(2)

print("\n--- Sample of Dynamic Pricing Recommendations ---")
print(results_df.head(10))
