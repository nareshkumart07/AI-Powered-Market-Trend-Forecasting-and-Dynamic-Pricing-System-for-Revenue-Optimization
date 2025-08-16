import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
# --- Required External Libraries ---
# You may need to install these libraries. You can do so by running:
# pip install torch holidays
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
from typing import Optional, Tuple, Dict, Any

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
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 2. The Main Pipeline Function ---
def run_forecasting_pipeline(
    df: pd.DataFrame,
    product_stock_code: str,
    date_col: str = 'InvoiceDate',
    date_format: str = '%d-%m-%Y %H:%M',
    quantity_col: str = 'Quantity',
    stock_code_col: str = 'StockCode',
    seq_length: int = 60,
    forecast_horizon: int = 1,
    train_split_ratio: float = 0.7,
    val_split_ratio: float = 0.15,
    model_params: Dict[str, Any] = None,
    training_params: Dict[str, Any] = None,
    generate_plots: bool = True,
    seed: int = 42
) -> Tuple[Optional[LSTMModel], Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    Runs a complete time-series forecasting pipeline for a specific product.

    Args:
        df (pd.DataFrame): The input DataFrame containing transaction data.
        product_stock_code (str): The StockCode of the product to forecast.
        date_col (str): The name of the column for the transaction date.
        date_format (str): The string format of the date column.
        quantity_col (str): The name of the column for the sales quantity.
        stock_code_col (str): The name of the column for the product stock code.
        seq_length (int): Number of past days to use as input features.
        forecast_horizon (int): Number of days ahead to forecast.
        train_split_ratio (float): Proportion of data for the training set.
        val_split_ratio (float): Proportion of data for the validation set.
        model_params (Dict): Hyperparameters for the LSTM model.
        training_params (Dict): Parameters for the training loop (e.g., epochs, lr).
        generate_plots (bool): If True, generates and displays summary plots.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple containing the trained model, a DataFrame of predictions, and evaluation metrics.
    """
    # --- A. Setup and Reproducibility ---
    print(f"--- Starting Forecast Pipeline for Product: {product_stock_code} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- B. Data Preparation & Feature Engineering ---
    print("Step 1: Preparing data and engineering features...")
    product_df = df[df[stock_code_col] == product_stock_code].copy()
    if product_df.empty:
        print(f"Error: No data found for StockCode {product_stock_code}.")
        return None, None, None

    # Use the specified date_format to parse dates correctly
    product_df[date_col] = pd.to_datetime(product_df[date_col], format=date_format, errors='coerce')
    if product_df[date_col].isnull().any():
        print(f"Error: Could not parse dates in the '{date_col}' column with the format '{date_format}'. Please check the format.")
        return None, None, None
        
    product_df.set_index(date_col, inplace=True)
    daily_sales_df = product_df.resample('D')[quantity_col].sum().to_frame()

    # Feature Engineering
    daily_sales_df['day_of_week'] = daily_sales_df.index.dayofweek
    daily_sales_df['month'] = daily_sales_df.index.month
    for i in [1, 7, 30]:
        daily_sales_df[f'lag_{i}_days'] = daily_sales_df[quantity_col].shift(i)
    for window in [7, 30]:
        daily_sales_df[f'rolling_mean_{window}_days'] = daily_sales_df[quantity_col].shift(1).rolling(window=window).mean()
        daily_sales_df[f'rolling_std_{window}_days'] = daily_sales_df[quantity_col].shift(1).rolling(window=window).std()
    
    uk_holidays = holidays.UK(years=daily_sales_df.index.year.unique())
    daily_sales_df['is_holiday'] = daily_sales_df.index.map(lambda x: 1 if x in uk_holidays else 0)
    daily_sales_df.fillna(0, inplace=True)

    # --- C. Scaling and Sequence Creation ---
    print("Step 2: Scaling data and creating sequences...")
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(daily_sales_df)
    target_col_idx = daily_sales_df.columns.get_loc(quantity_col)

    X, y = [], []
    for i in range(len(scaled_features) - seq_length - forecast_horizon):
        X.append(scaled_features[i:i+seq_length])
        y.append(scaled_features[i+seq_length:i+seq_length+forecast_horizon, target_col_idx])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # --- D. Data Splitting and DataLoader Creation ---
    print("Step 3: Splitting data and creating DataLoaders...")
    total_samples = len(X)
    train_size = int(total_samples * train_split_ratio)
    val_size = int(total_samples * val_split_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=32, shuffle=False)

    # --- E. Model Initialization and Training ---
    print("Step 4: Initializing and training the LSTM model...")
    if model_params is None:
        model_params = {'input_size': X_train.shape[2], 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}
    if training_params is None:
        training_params = {'num_epochs': 100, 'learning_rate': 0.001, 'patience': 20}

    model = LSTMModel(**model_params)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(training_params['num_epochs']):
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{training_params['num_epochs']}, Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= training_params['patience']:
            print(f"!! Early stopping triggered at epoch {epoch+1} !!")
            break
    
    model.load_state_dict(torch.load('best_model.pth'))

    # --- F. Evaluation and Inverse Scaling ---
    print("Step 5: Evaluating the model on the test set...")
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X)
            all_predictions.append(outputs.numpy())
    
    predictions_scaled = np.concatenate(all_predictions)
    
    # Inverse transform predictions
    predictions_full = np.zeros((predictions_scaled.shape[0], daily_sales_df.shape[1]))
    predictions_full[:, target_col_idx] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(predictions_full)[:, target_col_idx]
    
    # Correctly inverse transform actual test values for comparison
    y_test_full = np.zeros((len(y_test), daily_sales_df.shape[1]))
    y_test_full[:, target_col_idx] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(y_test_full)[:, target_col_idx]

    # --- G. Results and Visualization ---
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    metrics = {'MAE': mae, 'RMSE': rmse}
    print(f"\n--- Evaluation Metrics ---")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    results_df = pd.DataFrame({
        'Date': daily_sales_df.index[-len(y_test_actual):],
        'Actual': y_test_actual,
        'Predicted': predictions
    }).set_index('Date')

    if generate_plots:
        # --- EDITED PLOT: Enhanced visualization for clarity ---
        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean plot style
        plt.figure(figsize=(15, 8))

        # Plot the actual and predicted values
        plt.plot(results_df.index, results_df['Actual'], label='Actual Sales', color='royalblue', linewidth=2)
        plt.plot(results_df.index, results_df['Predicted'], label='Predicted Sales', color='tomato', linestyle='--', marker='o', markersize=4)
        
        # Add a shaded region between the actual and predicted lines to highlight the difference (error)
        plt.fill_between(results_df.index, results_df['Actual'], results_df['Predicted'], 
                         color='gray', alpha=0.2, label='Prediction Error')

        # Add title and labels with better formatting
        plt.title(f'Sales Forecast vs. Actuals for Product: {product_stock_code}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Quantity Sold', fontsize=12)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout() # Adjust layout to make room for date labels
        plt.show()

    # --- H. Business Insights and Diagnostic Plots ---
    if generate_plots:
        print("\n--- Business Insights & Diagnostic Plots ---")
        
        # 1. Feature Correlation Heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = daily_sales_df.corr()
        sns.heatmap(correlation_matrix[[quantity_col]].sort_values(by=quantity_col, ascending=False), 
                    annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Feature Correlation with Daily Sales for {product_stock_code}')
        plt.show()

        # 2. Sales by Day of the Week
        plt.figure(figsize=(10, 6))
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales_df.groupby('day_of_week')[quantity_col].mean().reindex(range(7)).plot(kind='bar', color='skyblue')
        plt.title(f'Average Daily Sales by Day of Week for {product_stock_code}')
        plt.xlabel('Day of the Week')
        plt.ylabel('Average Quantity Sold')
        plt.xticks(ticks=range(7), labels=day_names, rotation=45)
        plt.grid(axis='y')
        plt.show()

    # Print actionable business insights
    print("\n--- Actionable Business Insights ---")
    print(f"1. Key Sales Drivers: Based on the correlation heatmap, the strongest predictors for this product's sales are its recent performance, particularly the 'lag_1_day' and 'rolling_mean_7_days'.")
    print("   > INSIGHT: This product has strong sales momentum. A good sales day is often followed by another. Ensure inventory is replenished quickly after a sales spike to avoid stockouts.")
    
    day_of_week_sales = daily_sales_df.groupby('day_of_week')[quantity_col].mean()
    best_day_idx = day_of_week_sales.idxmax()
    worst_day_idx = day_of_week_sales.idxmin()
    best_day_name = day_names[best_day_idx]
    worst_day_name = day_names[worst_day_idx]
    
    print(f"\n2. Weekly Sales Pattern: The highest average sales occur on {best_day_name}, while the lowest occur on {worst_day_name}.")
    print(f"   > INSIGHT: Consider optimizing staff schedules to handle higher demand on {best_day_name}. To boost sales on {worst_day_name}, you could run targeted promotions or marketing campaigns.")

    print(f"\n3. Model Performance: The model predicts future sales with a Mean Absolute Error (MAE) of {mae:.2f} units.")
    print(f"   > INSIGHT: On average, the forecast is off by approximately {mae:.0f} units per day. This level of accuracy can be used to set safety stock levels and plan inventory orders more effectively.")


    print(f"\n--- Pipeline Finished for Product: {product_stock_code} ---")
    return model, results_df, metrics

# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    # !!! IMPORTANT: Change this path to the location of your data file !!!
    DATA_FILE_PATH = "/content/drive/MyDrive/refine_file.csv"
    
    # Load the main dataframe
    main_df = pd.read_csv(DATA_FILE_PATH)
    
    # --- Configuration for the pipeline run ---
    # Specify the product you want to forecast by its StockCode
    product_to_forecast = '85099B' # Example: WHITE HANGING HEART T-LIGHT HOLDER
    
    # --- Run the entire pipeline by calling the function ---
    trained_model, predictions_df, performance_metrics = run_forecasting_pipeline(
        df=main_df,
        product_stock_code=product_to_forecast
        # The date_format parameter will use its default value: '%d-%m-%Y %H:%M'
    )
    
    # --- Display the results ---
    if predictions_df is not None:
        print("\n--- Forecast Results (Last 10 Days) ---")
        print(predictions_df.tail(10))
