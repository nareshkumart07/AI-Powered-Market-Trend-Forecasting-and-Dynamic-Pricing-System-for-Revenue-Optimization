"""
This script performs a complete time-series forecasting analysis for a specific
product using a GRU (Gated Recurrent Unit) model. The code is structured into
modular, reusable functions for each step of the pipeline.
"""

# --- 0. LIBRARY IMPORTS ---
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
from typing import Optional, Tuple, Dict, Any

# --- STEP 1: MODEL ARCHITECTURE DEFINITION ---

class GRUModel(nn.Module):
    """Defines the structure of the GRU neural network."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# --- STEP 2: DATA PREPARATION AND FEATURE ENGINEERING ---

def prepare_data_for_product(
    df: pd.DataFrame, product_stock_code: str, date_col: str, date_format: str,
    quantity_col: str, stock_code_col: str
) -> Optional[pd.DataFrame]:
    """Filters data for a specific product and resamples it to a daily frequency."""
    print("--- Step 2a: Preparing and Resampling Data ---")
    product_df = df[df[stock_code_col] == product_stock_code].copy()
    if product_df.empty:
        print(f"Error: No data found for StockCode {product_stock_code}.")
        return None

    product_df[date_col] = pd.to_datetime(product_df[date_col], format=date_format, errors='coerce')
    if product_df[date_col].isnull().any():
        print(f"Error: Could not parse dates in '{date_col}'. Please check the format.")
        return None
        
    product_df.set_index(date_col, inplace=True)
    daily_sales_df = product_df.resample('D')[quantity_col].sum().to_frame()
    return daily_sales_df

def engineer_features(daily_df: pd.DataFrame, quantity_col: str) -> pd.DataFrame:
    """Engineers time-series features like lags, rolling stats, and holiday flags."""
    print("--- Step 2b: Engineering Features ---")
    df_featured = daily_df.copy()
    df_featured['day_of_week'] = df_featured.index.dayofweek
    df_featured['month'] = df_featured.index.month
    for i in [1, 7, 30]:
        df_featured[f'lag_{i}_days'] = df_featured[quantity_col].shift(i)
    for window in [7, 30]:
        df_featured[f'rolling_mean_{window}_days'] = df_featured[quantity_col].shift(1).rolling(window=window).mean()
        df_featured[f'rolling_std_{window}_days'] = df_featured[quantity_col].shift(1).rolling(window=window).std()
    
    uk_holidays = holidays.UK(years=df_featured.index.year.unique())
    df_featured['is_holiday'] = df_featured.index.map(lambda x: 1 if x in uk_holidays else 0)
    df_featured.fillna(0, inplace=True)
    return df_featured

# --- STEP 3: DATA PROCESSING FOR PYTORCH ---

def scale_and_create_sequences(
    daily_df: pd.DataFrame, quantity_col: str, seq_length: int, forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, int]:
    """Scales the data and creates input/output sequences for the model."""
    print("--- Step 3a: Scaling data and creating sequences ---")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(daily_df)
    target_col_idx = daily_df.columns.get_loc(quantity_col)

    X, y = [], []
    for i in range(len(scaled_features) - seq_length - forecast_horizon):
        X.append(scaled_features[i:i+seq_length])
        y.append(scaled_features[i+seq_length:i+seq_length+forecast_horizon, target_col_idx])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler, target_col_idx

def split_data_and_create_loaders(
    X: np.ndarray, y: np.ndarray, train_split_ratio: float, val_split_ratio: float
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Splits data into train, validation, and test sets and creates PyTorch DataLoaders."""
    print("--- Step 3b: Splitting data and creating DataLoaders ---")
    total_samples = len(X)
    train_size = int(total_samples * train_split_ratio)
    val_size = int(total_samples * val_split_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_test

# --- STEP 4: MODEL TRAINING ---

def train_gru_model(
    train_loader: DataLoader, val_loader: DataLoader, model_params: Dict, training_params: Dict
) -> Tuple[GRUModel, Dict]:
    """Initializes and trains the GRU model with early stopping."""
    print("--- Step 4: Initializing and training the GRU model ---")
    model = GRUModel(**model_params)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(training_params['num_epochs']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gru_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{training_params['num_epochs']}, Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= training_params['patience']:
            print(f"!! Early stopping triggered at epoch {epoch+1} !!")
            break
    
    model.load_state_dict(torch.load('best_gru_model.pth'))
    print("-> Model training complete.")
    return model, history

# --- STEP 5: EVALUATION AND REPORTING ---

def evaluate_model(
    model: GRUModel, test_loader: DataLoader, scaler: MinMaxScaler, y_test: np.ndarray,
    target_col_idx: int, num_features: int
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluates the model on the test set and returns predictions and metrics."""
    print("--- Step 5: Evaluating the model on the test set ---")
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X)
            all_predictions.append(outputs.numpy())
    
    predictions_scaled = np.concatenate(all_predictions)
    
    # Inverse transform predictions
    predictions_full = np.zeros((predictions_scaled.shape[0], num_features))
    predictions_full[:, target_col_idx] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(predictions_full)[:, target_col_idx]
    
    # Inverse transform actuals for comparison
    y_test_full = np.zeros((len(y_test), num_features))
    y_test_full[:, target_col_idx] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(y_test_full)[:, target_col_idx]

    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    metrics = {'MAE': mae, 'RMSE': rmse}
    
    results_df = pd.DataFrame({'Actual': y_test_actual, 'Predicted': predictions})
    return results_df, metrics

# --- STEP 6: VISUALIZATION AND INSIGHTS ---

def plot_training_history(history: Dict):
    """Plots the training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_forecast_vs_actuals(results_df: pd.DataFrame, daily_sales_index: pd.Index, product_stock_code: str):
    """Plots the final forecast against the actual sales data."""
    print("--- Step 6a: Plotting forecast results ---")
    results_df.index = daily_sales_index[-len(results_df):]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 8))
    plt.plot(results_df.index, results_df['Actual'], label='Actual Sales', color='royalblue', linewidth=2)
    plt.plot(results_df.index, results_df['Predicted'], label='Predicted Sales', color='tomato', linestyle='--', marker='o', markersize=4)
    plt.fill_between(results_df.index, results_df['Actual'], results_df['Predicted'], color='gray', alpha=0.2, label='Prediction Error')
    plt.title(f'GRU Sales Forecast vs. Actuals for Product: {product_stock_code}', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12); plt.ylabel('Quantity Sold', fontsize=12)
    plt.legend(fontsize=12); plt.xticks(rotation=45); plt.tight_layout()
    plt.show()

def generate_insights_and_diagnostics(daily_df: pd.DataFrame, quantity_col: str, product_stock_code: str, metrics: Dict):
    """Generates diagnostic plots and prints actionable business insights."""
    print("--- Step 6b: Generating insights and diagnostic plots ---")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Print insights
    print("\n--- Actionable Business Insights ---")
    print(f"1. Model Performance: The GRU model forecasts sales with a Mean Absolute Error (MAE) of {metrics['MAE']:.2f} units.")
    print(f"   > INSIGHT: On average, the forecast is off by approximately {metrics['MAE']:.0f} units per day. Use this to set safety stock levels.")
    
    day_of_week_sales = daily_df.groupby('day_of_week')[quantity_col].mean()
    best_day_name = day_names[day_of_week_sales.idxmax()]
    worst_day_name = day_names[day_of_week_sales.idxmin()]
    print(f"\n2. Weekly Sales Pattern: Highest sales on {best_day_name}, lowest on {worst_day_name}.")
    print(f"   > INSIGHT: Optimize staffing for {best_day_name}. Consider promotions to boost sales on {worst_day_name}.")

    # Plot diagnostics
    plt.figure(figsize=(10, 6))
    day_of_week_sales.reindex(range(7)).plot(kind='bar', color='skyblue')
    plt.title(f'Average Daily Sales by Day of Week for {product_stock_code}')
    plt.xlabel('Day of the Week'); plt.ylabel('Average Quantity Sold')
    plt.xticks(ticks=range(7), labels=day_names, rotation=45); plt.grid(axis='y')
    plt.show()

# --- 7. MAIN PIPELINE ORCHESTRATOR ---

def run_gru_forecasting_pipeline(
    df: pd.DataFrame, product_stock_code: str, date_col: str = 'InvoiceDate',
    date_format: str = '%d-%m-%Y %H:%M', quantity_col: str = 'Quantity',
    stock_code_col: str = 'StockCode', seq_length: int = 60, forecast_horizon: int = 1,
    train_split_ratio: float = 0.7, val_split_ratio: float = 0.15,
    model_params: Dict = None, training_params: Dict = None, seed: int = 42
) -> Tuple[Optional[GRUModel], Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """Orchestrates the complete time-series forecasting pipeline."""
    print(f"--- Starting GRU Forecast Pipeline for Product: {product_stock_code} ---")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Step 2: Data Prep
    daily_sales_df = prepare_data_for_product(df, product_stock_code, date_col, date_format, quantity_col, stock_code_col)
    if daily_sales_df is None: return None, None, None
    daily_sales_df = engineer_features(daily_sales_df, quantity_col)

    # Step 3: Data Processing
    X, y, scaler, target_col_idx = scale_and_create_sequences(daily_sales_df, quantity_col, seq_length, forecast_horizon)
    train_loader, val_loader, test_loader, y_test = split_data_and_create_loaders(X, y, train_split_ratio, val_split_ratio)

    # Step 4: Model Training
    if model_params is None:
        model_params = {'input_size': X.shape[2], 'hidden_size': 128, 'num_layers': 2, 'output_size': forecast_horizon}
    if training_params is None:
        training_params = {'num_epochs': 100, 'learning_rate': 0.001, 'patience': 25}
    model, history = train_gru_model(train_loader, val_loader, model_params, training_params)

    # Step 5 & 6: Evaluation and Reporting
    results_df, metrics = evaluate_model(model, test_loader, scaler, y_test, target_col_idx, daily_sales_df.shape[1])
    
    plot_training_history(history)
    plot_forecast_vs_actuals(results_df, daily_sales_df.index, product_stock_code)
    generate_insights_and_diagnostics(daily_sales_df, quantity_col, product_stock_code, metrics)

    print(f"\n--- Pipeline Finished for Product: {product_stock_code} ---")
    return model, results_df, metrics

# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    try:
        DATA_FILE_PATH = "/content/refine_file.csv"
        main_df = pd.read_csv(DATA_FILE_PATH)
        product_to_forecast = '85099B'
        
        trained_model, predictions_df, performance_metrics = run_gru_forecasting_pipeline(
            df=main_df,
            product_stock_code=product_to_forecast
        )
        
        if predictions_df is not None:
            print("\n--- GRU Forecast Results (Last 10 Days) ---")
            print(predictions_df.tail(10))
    except FileNotFoundError:
        print(f"Error: The file at '{DATA_FILE_PATH}' was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
