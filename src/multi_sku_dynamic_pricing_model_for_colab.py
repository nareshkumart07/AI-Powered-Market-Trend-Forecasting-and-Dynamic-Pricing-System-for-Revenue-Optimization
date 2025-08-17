import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import holidays
import random
import warnings
from typing import List, Dict, Any, Optional, Tuple

# Suppress harmless warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. LSTM Model Architecture ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- 2. Dynamic Pricing Calculation Function ---
def calculate_dynamic_price(base_price, forecasted_demand, baseline_demand, elasticity=-1.5, intensity=0.1):
    if baseline_demand == 0: return base_price # Avoid division by zero
    demand_diff = (forecasted_demand - baseline_demand) / baseline_demand
    price_change_factor = (demand_diff / elasticity) * intensity
    new_price = base_price * (1 + price_change_factor)
    # Apply price guardrails (e.g., +/- 25% of base price)
    max_price, min_price = base_price * 1.25, base_price * 0.75
    return max(min(new_price, max_price), min_price)

# --- 3. The Main Pipeline Function ---
def run_multi_sku_pipeline(
    sales_df: pd.DataFrame,
    competitor_df: pd.DataFrame,
    customer_segments_df: pd.DataFrame, # <-- NEW: Added customer segments
    product_ids: List[str],
    seq_length: int = 30,
    seed: int = 42
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Runs a complete forecasting and dynamic pricing pipeline for a list of products.

    Args:
        sales_df (pd.DataFrame): DataFrame with historical sales data.
        competitor_df (pd.DataFrame): DataFrame with competitor pricing.
        customer_segments_df (pd.DataFrame): DataFrame with customer segment data.
        product_ids (List[str]): A list of StockCodes to analyze.
        seq_length (int): The number of past days to use for forecasting.
        seed (int): Random seed for reproducibility.

    Returns:
        A tuple containing a DataFrame of price recommendations and a dictionary
        with data for visualizing the top product's forecast.
    """
    # --- A. Setup ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    all_results = []
    top_product_viz_data = None

    # --- B. Main Loop for Each SKU ---
    for i, product_id in enumerate(product_ids):
        print(f"\n{'='*60}\n({i+1}/{len(product_ids)}) Processing Product (SKU): {product_id}\n{'='*60}")

        # --- Data Preparation & Feature Engineering ---
        # NEW: Merge sales data with customer segments
        sales_df_merged = pd.merge(sales_df, customer_segments_df, on='Customer ID', how='left')
        
        product_df = sales_df_merged[sales_df_merged['StockCode'] == product_id].copy()
        daily_sales_df = product_df.resample('D', on='InvoiceDate')['Quantity'].sum().to_frame()

        # NEW: Engineer features from customer segments
        if 'Segment' in product_df.columns:
            segment_dummies = pd.get_dummies(product_df['Segment'])
            daily_segments = segment_dummies.resample('D', on='InvoiceDate').sum()
            daily_sales_df = pd.concat([daily_sales_df, daily_segments], axis=1)

        # Standard Feature Engineering
        daily_sales_df['day_of_week'] = daily_sales_df.index.dayofweek
        daily_sales_df['month'] = daily_sales_df.index.month
        daily_sales_df['lag_7_days'] = daily_sales_df['Quantity'].shift(7)
        daily_sales_df['rolling_mean_7'] = daily_sales_df['Quantity'].shift(1).rolling(window=7).mean()

        product_competitor_prices = competitor_df[competitor_df['StockCode'] == product_id]
        if not product_competitor_prices.empty:
            for col in ['Competitor_A_Price', 'Competitor_B_Price', 'Competitor_C_Price']:
                daily_sales_df[col.lower()] = product_competitor_prices[col].iloc[0]
        daily_sales_df.fillna(0, inplace=True)

        # --- Scaling and Sequencing ---
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(daily_sales_df)
        target_col_idx = daily_sales_df.columns.get_loc("Quantity")
        
        X, y = [], []
        for j in range(len(scaled_features) - seq_length):
            X.append(scaled_features[j:j + seq_length])
            y.append(scaled_features[j + seq_length, target_col_idx])
        X, y = np.array(X), np.array(y)

        if len(X) < 20:
            print(f"Skipping {product_id} due to insufficient historical data.")
            continue

        # --- Data Splitting and DataLoader ---
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test_actual_values = X[train_size:], daily_sales_df['Quantity'][-len(X) + train_size:].values
        
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
        X_test_tensor = torch.from_numpy(X_test).float()
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=False)

        # --- Model Training ---
        model = LSTMModel(input_size=X_train.shape[2], hidden_size=50, num_layers=2, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        print(f"Training demand forecast model for {product_id}...")
        for epoch in range(50):
            for batch_X, batch_y in train_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # --- Forecasting and Dynamic Pricing ---
        model.eval()
        last_sequence = torch.from_numpy(scaled_features[-seq_length:]).unsqueeze(0).float()
        with torch.no_grad():
            next_day_scaled_pred = model(last_sequence).item()

        dummy_pred = np.zeros((1, scaled_features.shape[1])); dummy_pred[:, target_col_idx] = next_day_scaled_pred
        next_day_forecast = scaler.inverse_transform(dummy_pred)[:, target_col_idx][0]
        next_day_forecast = max(0, next_day_forecast)

        BASE_PRICE = product_df['Price'].mean()
        BASELINE_DEMAND = daily_sales_df['Quantity'].mean()
        suggested_price = calculate_dynamic_price(BASE_PRICE, next_day_forecast, BASELINE_DEMAND)
        
        if suggested_price > BASE_PRICE * 1.02:
            recommendation = "Increase Price 📈"
        elif suggested_price < BASE_PRICE * 0.98:
            recommendation = "Promotional Price 📉"
        else:
            recommendation = "Hold Price ⏸️"

        # --- Store Results ---
        all_results.append({
            'StockCode': product_id,
            'Base_Price': f"${BASE_PRICE:,.2f}",
            'Avg_Daily_Demand': f"{BASELINE_DEMAND:,.0f}",
            'Forecasted_Demand_Next_Day': f"{next_day_forecast:,.0f}",
            'Suggested_Dynamic_Price': f"${suggested_price:,.2f}",
            'Recommendation': recommendation
        })

        # --- Store data for visualization for the first product ---
        if i == 0:
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
            predictions_scaled = test_outputs.numpy().flatten()
            dummy_array = np.zeros((len(predictions_scaled), scaled_features.shape[1]))
            dummy_array[:, target_col_idx] = predictions_scaled
            forecasted_demand_viz = np.maximum(0, scaler.inverse_transform(dummy_array)[:, target_col_idx])
            
            top_product_viz_data = {
                'dates': daily_sales_df.index[-len(X_test):],
                'actuals': y_test_actual_values,
                'forecasts': forecasted_demand_viz,
                'product_id': product_id,
                'base_price': BASE_PRICE,
                'baseline_demand': BASELINE_DEMAND,
                'next_day_forecast': next_day_forecast,
                'competitor_prices': product_competitor_prices
            }

    return pd.DataFrame(all_results), top_product_viz_data

# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    # --- Configuration ---
    SALES_DATA_PATH = "/content/drive/MyDrive/refine_file.csv"
    COMPETITOR_DATA_PATH = "competitor_prices.csv"
    SEGMENTS_DATA_PATH = "/content/customer_segmentation_kmeans_results.csv" # <-- NEW: Path to segment data

    # --- Load Data ---
    try:
        sales_df = pd.read_csv(SALES_DATA_PATH)
        competitor_df = pd.read_csv(COMPETITOR_DATA_PATH)
        segments_df = pd.read_csv(SEGMENTS_DATA_PATH) # <-- NEW: Load segment data
        print("Successfully loaded sales, competitor, and customer segment data.")
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find a required file: {e.filename}")
        print("Please make sure all three CSV files are in the correct location.")
        exit()

    sales_df['InvoiceDate'] = pd.to_datetime(sales_df['InvoiceDate'], errors='coerce')

    # --- Identify Top 5 Products ---
    top_5_products = sales_df['StockCode'].value_counts().nlargest(5).index.tolist()
    print(f"\nIdentified Top 5 Products for Analysis: {top_5_products}")

    # --- Run Pipeline ---
    summary_df, viz_data = run_multi_sku_pipeline(
        sales_df=sales_df,
        competitor_df=competitor_df,
        customer_segments_df=segments_df, # <-- NEW: Pass segment data to function
        product_ids=top_5_products
    )

    # --- Final Summary ---
    print(f"\n{'='*60}\nDynamic Pricing Recommendations Summary\n{'='*60}")
    if not summary_df.empty:
        print(summary_df.to_string())
    else:
        print("No results were generated. This might be due to insufficient data for the top products.")

    # --- Visualization and Business Insights for the Top Product ---
    if viz_data:
        print(f"\n\n--- 📈 Business Insights for Top Product: {viz_data['product_id']} ---")
        
        # --- Plot 1: Demand Forecast vs. Actuals ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(15, 7))
        ax1.plot(viz_data['dates'], viz_data['actuals'], label='Actual Demand', color='royalblue', linestyle='--', alpha=0.8)
        ax1.plot(viz_data['dates'], viz_data['forecasts'], label='Forecasted Demand', color='tomato', linewidth=2, marker='o', markersize=4)
        ax1.set_title(f"Demand Forecast vs. Actuals for SKU: {viz_data['product_id']}", fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Quantity Sold', fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # --- Plot 2: Revenue Optimization Curve ---
        price_points = np.linspace(viz_data['base_price'] * 0.75, viz_data['base_price'] * 1.25, 20)
        expected_revenues = []
        for price in price_points:
            price_diff_ratio = (price - viz_data['base_price']) / viz_data['base_price']
            demand_diff_ratio = price_diff_ratio * -1.5 # Elasticity
            expected_demand = viz_data['baseline_demand'] * (1 + demand_diff_ratio)
            expected_revenues.append(price * expected_demand)
        
        optimal_price_index = np.argmax(expected_revenues)
        optimal_price = price_points[optimal_price_index]
        max_revenue = expected_revenues[optimal_price_index]

        plt.figure(figsize=(12, 7))
        plt.plot(price_points, expected_revenues, marker='o', linestyle='-', color='darkgreen')
        plt.axvline(x=optimal_price, color='red', linestyle='--', label=f'Optimal Price: ${optimal_price:,.2f}')
        plt.title('Revenue Optimization Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Price Point ($)', fontsize=12)
        plt.ylabel('Expected Daily Revenue ($)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # --- Actionable Insights ---
        print("\n--- 📝 Actionable Business Insights ---")
        print(f"1. **Inventory Management**: The forecast for {viz_data['product_id']} has a Mean Absolute Error of {mean_absolute_error(viz_data['actuals'], viz_data['forecasts']):.2f} units. Use this to set your safety stock. Keeping an extra **{np.ceil(mean_absolute_error(viz_data['actuals'], viz_data['forecasts'])):.0f} units** on hand can prevent stockouts.")
        print(f"2. **Revenue Maximization**: The optimization curve shows that a price of **${optimal_price:,.2f}** is projected to yield the maximum daily revenue of **${max_revenue:,.2f}**. This is a data-driven starting point for A/B testing your prices.")
        print(f"3. **Customer-Driven Demand**: The model now understands how different customer segments ('Champions', 'At Risk', etc.) impact sales. If a high forecast is driven by 'At-Risk' customers, the marketing team can launch a targeted re-engagement campaign. If it's driven by 'New Customers', focus on welcome offers.")
        print(f"4. **Market Positioning**: The model uses competitor prices as a feature. If your forecast accuracy drops, it could be a sign that a competitor has made a significant price change, providing a valuable, early insight for your pricing strategy team.")
