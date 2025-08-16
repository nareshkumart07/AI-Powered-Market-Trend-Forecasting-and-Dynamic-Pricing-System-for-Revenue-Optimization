import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
from datetime import timedelta

# --- Required External Libraries ---
# You may need to install these libraries. You can do so by running:
# pip install torch holidays
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
from typing import Optional, Tuple, Dict, Any, List

# --- 1. Define the Model Architecture ---
# This class defines the structure of the LSTM model.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # We pass both states to the LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

# --- 2. The Main Pipeline Function ---
def run_dynamic_pricing_pipeline(
    df: pd.DataFrame,
    customer_segments_df: pd.DataFrame,
    product_stock_code: str, # <-- EDITED: Now takes a single product code
    date_col: str = 'InvoiceDate',
    date_format: str = '%d-%m-%Y %H:%M',
    quantity_col: str = 'Quantity',
    price_col: str = 'Price',
    stock_code_col: str = 'StockCode',
    customer_id_col: str = 'Customer ID',
    seq_length: int = 30,
    seed: int = 42
) -> None:
    """
    Runs a forecasting and dynamic pricing pipeline for a single product.
    """
    # --- A. Setup and Reproducibility ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"\n--- Starting Pipeline for Product: {product_stock_code} ---")

    # --- B. Data Preparation & Feature Engineering ---
    print("Step 1: Preparing data and engineering features...")
    df_merged = pd.merge(df.reset_index(), customer_segments_df, on=customer_id_col, how='left')
    df_merged[date_col] = pd.to_datetime(df_merged[date_col], format=date_format, errors='coerce')
    df_merged.set_index(date_col, inplace=True)
    
    product_df = df_merged[df_merged[stock_code_col] == product_stock_code].copy()
    if product_df.empty:
        print(f"Error: No data found for StockCode {product_stock_code}. Exiting.")
        return

    daily_data = product_df.resample('D').agg({quantity_col: 'sum', price_col: 'mean'})
    segment_dummies = pd.get_dummies(product_df['Segment'])
    daily_segments = segment_dummies.resample('D').sum()
    daily_data = pd.concat([daily_data, daily_segments], axis=1)
    daily_data['day_of_week'] = daily_data.index.dayofweek
    daily_data['month'] = daily_data.index.month
    daily_data['lag_7_days'] = daily_data[quantity_col].shift(7)
    daily_data['rolling_mean_7_days'] = daily_data[quantity_col].shift(1).rolling(window=7).mean()
    daily_data.fillna(0, inplace=True)

    # --- Scaling and Sequence Creation ---
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(daily_data)
    target_col_idx = daily_data.columns.get_loc(quantity_col)

    X, y = [], []
    for j in range(len(scaled_features) - seq_length):
        X.append(scaled_features[j:j + seq_length])
        y.append(scaled_features[j + seq_length, target_col_idx])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    if len(X) < 50:
        print(f"Warning: Not enough historical data for {product_stock_code} to train a model. Exiting.")
        return

    # --- Data Splitting ---
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train).view(-1, 1)
    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val).view(-1, 1)

    # --- Grid Search (simplified for speed) ---
    print("Step 2: Finding best model hyperparameters...")
    param_grid = {'hidden_size': [50], 'num_layers': [2], 'learning_rate': [0.001], 'batch_size': [32]}
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    best_params = {}
    best_global_val_loss = float('inf')

    for params in all_params:
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=params['batch_size'], shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=params['batch_size'], shuffle=False)
        model = LSTMModel(input_size=X_train.shape[2], hidden_size=params['hidden_size'], num_layers=params['num_layers'], output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        for epoch in range(50):
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

        if avg_val_loss < best_global_val_loss:
            best_global_val_loss = avg_val_loss
            best_params = params
            torch.save(model.state_dict(), 'best_lstm_dynamic_price_model.pth')
    
    print(f"Best parameters for this product: {best_params}")

    # --- Final Model Evaluation ---
    print("Step 3: Evaluating the best model...")
    best_model = LSTMModel(input_size=X_train.shape[2], hidden_size=best_params['hidden_size'], num_layers=params['num_layers'], output_size=1)
    best_model.load_state_dict(torch.load('best_lstm_dynamic_price_model.pth'))
    best_model.eval()
    
    # --- Dynamic Pricing Logic ---
    print("Step 4: Calculating dynamic price recommendations...")
    def calculate_dynamic_price(base_price, forecasted_demand, baseline_demand, elasticity=-1.5, intensity=0.1):
        demand_diff_ratio = (forecasted_demand - baseline_demand) / baseline_demand
        price_change_factor = (demand_diff_ratio / elasticity) * intensity
        new_price = base_price * (1 + price_change_factor)
        return max(min(new_price, base_price * 1.25), base_price * 0.75)

    BASE_PRICE = daily_data[price_col].mean()
    BASELINE_DEMAND = daily_data[quantity_col].mean()

    # --- Detailed 10-Day Analysis ---
    print("Step 5: Generating detailed 10-day analysis...")
    # --- Historical Performance Table ---
    X_test_tensor = torch.from_numpy(X_test)
    with torch.no_grad():
        test_predictions_scaled = best_model(X_test_tensor).numpy().flatten()
    
    dummy_array = np.zeros((len(test_predictions_scaled), scaled_features.shape[1])); dummy_array[:, target_col_idx] = test_predictions_scaled
    forecasted_demand_test = scaler.inverse_transform(dummy_array)[:, target_col_idx]
    
    dummy_array_actual = np.zeros((len(y_test), scaled_features.shape[1])); dummy_array_actual[:, target_col_idx] = y_test
    actual_demand_test = scaler.inverse_transform(dummy_array_actual)[:, target_col_idx]
    
    test_dates = daily_data.index[-len(actual_demand_test):]
    suggested_prices_test = [calculate_dynamic_price(BASE_PRICE, f_demand, BASELINE_DEMAND) for f_demand in forecasted_demand_test]

    historical_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Demand': actual_demand_test,
        'Forecasted Demand': forecasted_demand_test,
        'Base Price': BASE_PRICE,
        'Price Recommendation': suggested_prices_test
    })
    
    historical_df['Date'] = historical_df['Date'].dt.strftime('%Y-%m-%d')
    for col in ['Base Price', 'Price Recommendation']:
        historical_df[col] = historical_df[col].apply(lambda x: f"${x:,.2f}")
    for col in ['Actual Demand', 'Forecasted Demand']:
        historical_df[col] = historical_df[col].round(0).astype(int)

    print("\n--- 📜 Historical Performance (Last 10 Days of Test Data) ---")
    print(historical_df.tail(10).to_string(index=False))

    # --- Future 10-Day Forecast ---
    future_forecasts = []
    current_sequence = scaled_features[-seq_length:]
    for _ in range(10):
        with torch.no_grad():
            pred_scaled = best_model(torch.from_numpy(current_sequence).unsqueeze(0).float()).item()
        future_forecasts.append(pred_scaled)
        new_row = current_sequence[-1].copy(); new_row[target_col_idx] = pred_scaled
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    dummy_array = np.zeros((len(future_forecasts), scaled_features.shape[1])); dummy_array[:, target_col_idx] = future_forecasts
    future_demand = scaler.inverse_transform(dummy_array)[:, target_col_idx]
    future_prices = [calculate_dynamic_price(BASE_PRICE, f_demand, BASELINE_DEMAND) for f_demand in future_demand]
    future_dates = [daily_data.index[-1] + timedelta(days=d) for d in range(1, 11)]

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Actual Demand': ['N/A'] * 10,
        'Forecasted Demand': future_demand,
        'Base Price': BASE_PRICE,
        'Price Recommendation': future_prices
    })

    future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
    future_df['Base Price'] = future_df['Base Price'].apply(lambda x: f"${x:,.2f}")
    future_df['Price Recommendation'] = future_df['Price Recommendation'].apply(lambda x: f"${x:,.2f}")
    future_df['Forecasted Demand'] = future_df['Forecasted Demand'].round(0).astype(int)
    future_df = future_df[['Date', 'Actual Demand', 'Forecasted Demand', 'Base Price', 'Price Recommendation']]

    print("\n--- 🔮 Future Forecast & Price Recommendations (Next 10 Days) ---")
    print(future_df.to_string(index=False))

    # --- Visualization ---
    # Create two subplots that share the same x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig.suptitle(f'Dynamic Pricing & Demand Forecast for Product: {product_stock_code}', fontsize=16, fontweight='bold')

    # --- Plot 1: Demand Forecast ---
    ax1.set_ylabel('Daily Quantity Sold', color='royalblue', fontsize=12)
    ax1.plot(test_dates, actual_demand_test, label='Actual Demand', color='royalblue', linestyle='--', alpha=0.7)
    ax1.plot(test_dates, forecasted_demand_test, label='Historical Forecast', color='darkblue', marker='o', markersize=4)
    ax1.plot(future_dates, future_demand, label='Future Forecast', color='green', marker='o', markersize=4, linestyle=':')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Add point numbers (annotations) to the demand forecast plot
    for date, val in zip(test_dates, forecasted_demand_test):
        ax1.text(date, val + 0.5, f'{val:.0f}', ha='center', va='bottom', fontsize=9, color='darkblue')
    for date, val in zip(future_dates, future_demand):
        ax1.text(date, val + 0.5, f'{val:.0f}', ha='center', va='bottom', fontsize=9, color='green')

    # --- Plot 2: Price Recommendation ---
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Suggested Price ($)', color='tomato', fontsize=12)
    ax2.plot(test_dates, suggested_prices_test, label='Historical Price Rec.', color='tomato', marker='o', markersize=4, linestyle='-')
    ax2.plot(future_dates, future_prices, label='Future Price Rec.', color='darkred', marker='x', linestyle=':')
    ax2.axhline(y=BASE_PRICE, color='gray', linestyle=':', label=f'Base Price (${BASE_PRICE:.2f})')
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # Add point numbers (annotations) to the price recommendation plot
    for date, val in zip(test_dates, suggested_prices_test):
        ax2.text(date, val + 0.01, f'${val:.2f}', ha='center', va='bottom', fontsize=9, color='tomato')
    for date, val in zip(future_dates, future_prices):
        ax2.text(date, val + 0.01, f'${val:.2f}', ha='center', va='bottom', fontsize=9, color='darkred')

    fig.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make room for suptitle
    plt.xticks(rotation=45)
    plt.show()


# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    # --- Configuration ---
    SALES_DATA_PATH = "/content/drive/MyDrive/refine_file.csv"
    SEGMENTS_DATA_PATH = "customer_segmentation_results.csv" # Assumes the RFM output is saved here
    PRODUCT_TO_FORECAST = '85099B' # Define the single product to analyze

    # --- Load Data ---
    try:
        sales_df = pd.read_csv(SALES_DATA_PATH)
        segments_df = pd.read_csv(SEGMENTS_DATA_PATH)
        
        # --- Run Pipeline for the Single Product ---
        run_dynamic_pricing_pipeline(
            df=sales_df,
            customer_segments_df=segments_df,
            product_stock_code=PRODUCT_TO_FORECAST
        )
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required data file: {e.filename}")
        print("Please ensure both the sales data and the customer segmentation CSV files are in the correct location.")
