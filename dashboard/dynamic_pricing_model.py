"""
This script performs a complete time-series forecasting and dynamic pricing analysis
for a specific product. It trains a separate, specialized model for each customer
segment to provide more accurate, granular forecasts and price recommendations.
"""

# --- 0. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import holidays
from typing import Optional, Tuple, Dict, List
import random
import os

# --- STEP 1: MODEL ARCHITECTURE DEFINITION ---
class LSTMModel(nn.Module):
    """Defines the structure of the LSTM neural network."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- STEP 2: SEGMENT FORECASTER AND PRICER CLASS ---
class SegmentForecaster:
    """
    Encapsulates the entire forecasting and dynamic pricing pipeline for a single customer segment.
    """
    def __init__(self, df: pd.DataFrame, segment: str, product_code: str, competitor_path: str, config: Dict):
        self.df = df
        self.segment = segment
        self.product_code = product_code
        self.competitor_path = competitor_path
        self.config = config
        self.daily_df = None
        self.model = None
        self.scaler = None
        self.target_col_idx = None

        torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])

    def _prepare_data(self) -> bool:
        """Filters, merges, and engineers features for the segment's data."""
        print(f"\n--- Preparing data for Segment: {self.segment} ---")

        segment_df = self.df[(self.df['StockCode'] == self.product_code) & (self.df['Cluster_Persona'] == self.segment)].copy()

        if segment_df.empty:
            print(f"Warning: No sales data found for segment '{self.segment}'. Skipping.")
            return False

        segment_df[self.config['date_col']] = pd.to_datetime(segment_df[self.config['date_col']], format=self.config['date_format'], errors='coerce')
        segment_df.set_index(self.config['date_col'], inplace=True)
        daily_sales = segment_df.resample('D')[self.config['quantity_col']].sum().to_frame()

        if len(daily_sales) < self.config['seq_length'] * 2:
            print(f"Warning: Insufficient historical data for segment '{self.segment}' ({len(daily_sales)} days). Skipping.")
            return False

        try:
            comp_df = pd.read_csv(self.competitor_path)
            comp_df['Date'] = pd.to_datetime(comp_df['Date'], errors='coerce')
            comp_df.set_index('Date', inplace=True)
            daily_comp_prices = comp_df.resample('D').mean()
            merged_df = daily_sales.merge(daily_comp_prices, left_index=True, right_index=True, how='left')
            for col in ['our_price', 'competitor_A', 'competitor_B', 'competitor_C']:
                 if col in merged_df.columns: # Check if column exists after merge
                    merged_df[col].ffill(inplace=True); merged_df[col].bfill(inplace=True)
        except FileNotFoundError:
            print(f"Warning: Competitor data file not found at '{self.competitor_path}'. Proceeding without it.")
            merged_df = daily_sales # Proceed with only sales data
        except Exception as e:
            print(f"Warning: Error merging competitor data: {e}. Proceeding without it.")
            merged_df = daily_sales


        merged_df['day_of_week'] = merged_df.index.dayofweek
        merged_df['month'] = merged_df.index.month
        for lag in [1, 7, 30]:
            merged_df[f'lag_{lag}'] = merged_df[self.config['quantity_col']].shift(lag)
        for window in [7, 30]:
            merged_df[f'rolling_mean_{window}'] = merged_df[self.config['quantity_col']].shift(1).rolling(window=window).mean()

        try:
            country = self.config.get('holiday_country', 'UK')
            years = list(pd.Series(merged_df.index.year).unique())
            country_holidays = getattr(holidays, country)(years=years)
            merged_df['is_holiday'] = merged_df.index.map(lambda x: 1 if x in country_holidays else 0)
        except Exception:
            merged_df['is_holiday'] = 0 # fallback if config country invalid or holidays library fails


        merged_df.fillna(0, inplace=True)

        self.daily_df = merged_df
        print(f"-> Data prepared successfully for '{self.segment}' with {len(self.daily_df)} daily records.")
        return True

    def _build_dataloaders(self) -> Tuple:
        """Scales data, creates sequences, and builds PyTorch DataLoaders."""
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(self.daily_df)
        self.target_col_idx = self.daily_df.columns.get_loc(self.config['quantity_col'])

        X, y = [], []
        for i in range(len(scaled_features) - self.config['seq_length'] - self.config['forecast_horizon']):
            X.append(scaled_features[i:i+self.config['seq_length']])
            y.append(scaled_features[i+self.config['seq_length']:i+self.config['seq_length']+self.config['forecast_horizon'], self.target_col_idx])

        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        train_size = int(len(X) * self.config['train_split'])
        val_size = int(len(X) * self.config['val_split'])

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=False)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=32, shuffle=False)

        return train_loader, val_loader, test_loader, y_test

    def _train(self, train_loader: DataLoader, val_loader: DataLoader, model_params: Dict) -> Dict:
        """Handles the model training loop."""
        print(f"--- Training model for Segment: {self.segment} ---")
        self.model = LSTMModel(**model_params)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        history = {'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        model_save_path = f"best_model_{self.segment.replace(' / ', '_').replace(' ', '_')}.pth"

        for epoch in range(self.config['epochs']):
            self.model.train()
            for batch_X, batch_y in train_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_loss += criterion(self.model(batch_X), batch_y).item()

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), model_save_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}, Val Loss: {avg_val_loss:.4f}")

            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        try:
            self.model.load_state_dict(torch.load(model_save_path))
            print(f"-> Model for '{self.segment}' loaded from '{model_save_path}'")
        except Exception:
            print(f"Warning: Could not load best model from '{model_save_path}'. Using final epoch weights.")

        return history

    def _evaluate(self, test_loader: DataLoader, y_test: np.ndarray) -> Tuple[pd.DataFrame, float]:
        """Evaluates the model and returns a results DataFrame and the MAE."""
        print(f"--- Evaluating model for Segment: {self.segment} ---")
        self.model.eval()
        preds_scaled = np.concatenate([self.model(batch_X).detach().numpy() for batch_X, _ in test_loader])

        num_features = self.daily_df.shape[1]
        preds_full = np.zeros((len(preds_scaled), num_features))
        preds_full[:, self.target_col_idx] = preds_scaled.flatten()
        predictions = self.scaler.inverse_transform(preds_full)[:, self.target_col_idx]

        y_test_full = np.zeros((len(y_test), num_features))
        y_test_full[:, self.target_col_idx] = y_test.flatten()
        y_test_actual = self.scaler.inverse_transform(y_test_full)[:, self.target_col_idx]

        mae = mean_absolute_error(y_test_actual, predictions)
        print(f"-> Test MAE for {self.segment}: {mae:.2f}")

        return pd.DataFrame({'Actual': y_test_actual, 'Predicted': predictions}), mae

    def _forecast_future(self) -> pd.DataFrame:
        """Generates a forecast for future days."""
        print(f"--- Forecasting future sales for Segment: {self.segment} ---")
        self.model.eval()

        last_sequence = self.scaler.transform(self.daily_df.tail(self.config['seq_length']))
        current_sequence = torch.from_numpy(last_sequence).unsqueeze(0).float()

        future_preds = []
        with torch.no_grad():
            for _ in range(self.config['future_forecast_days']):
                pred = self.model(current_sequence).item()
                future_preds.append(pred)

                new_row = current_sequence.numpy().squeeze()[-1]
                new_row[self.target_col_idx] = pred
                new_sequence = np.vstack([current_sequence.numpy().squeeze()[1:], new_row])
                current_sequence = torch.from_numpy(new_sequence).unsqueeze(0).float()

        num_features = self.daily_df.shape[1]
        future_preds_full = np.zeros((len(future_preds), num_features))
        future_preds_full[:, self.target_col_idx] = future_preds
        future_predictions = self.scaler.inverse_transform(future_preds_full)[:, self.target_col_idx]

        last_date = self.daily_df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=self.config['future_forecast_days'])

        return pd.DataFrame({'Date': future_dates, 'Future_Forecast': future_predictions}).set_index('Date')

    def _calculate_dynamic_prices(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates dynamic prices based on future demand forecasts."""
        print(f"--- Calculating dynamic prices for Segment: {self.segment} ---")

        # Ensure 'our_price' column exists before calculating mean
        if 'our_price' not in self.daily_df.columns:
            print("Warning: 'our_price' column not found. Cannot calculate dynamic prices.")
            future_df['Price_Recommendation'] = np.nan # Or some default price
            return future_df

        base_price = self.daily_df['our_price'].mean()
        baseline_demand = self.daily_df[self.config['quantity_col']].mean()

        if base_price == 0 or baseline_demand <= 0:
            print("Warning: Base price is zero or baseline demand is zero/negative. Using base price for all recommendations.")
            future_df['Price_Recommendation'] = base_price
            return future_df

        def price_logic(forecasted_demand):
            demand_ratio = (forecasted_demand - baseline_demand) / baseline_demand
            price_factor = (demand_ratio / self.config['elasticity']) * self.config['intensity']
            new_price = base_price * (1 + price_factor)
            return max(min(new_price, base_price * 1.25), base_price * 0.75)

        future_df['Price_Recommendation'] = future_df['Future_Forecast'].apply(price_logic)
        return future_df

    def _visualize(self, results_df: pd.DataFrame, future_df_with_prices: pd.DataFrame):
        """Generates all plots for the segment, including dynamic pricing."""
        print(f"--- Visualizing results for Segment: {self.segment} ---")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'Forecast & Pricing for Product: {self.product_code} (Segment: {self.segment})', fontsize=16, fontweight='bold')

        results_df.index = self.daily_df.index[-len(results_df):]
        ax1.plot(results_df.index, results_df['Actual'], label='Actual Sales', color='royalblue')
        ax1.plot(results_df.index, results_df['Predicted'], label='Predicted Sales (Test Set)', color='tomato', linestyle='--')
        ax1.plot(future_df_with_prices.index, future_df_with_prices['Future_Forecast'], label='Future Forecast', color='green', linestyle=':')
        ax1.set_ylabel('Quantity Sold'); ax1.legend(); ax1.grid(True)
        ax1.set_title('Sales Demand Forecast')

        # Add value labels to Demand Forecast plot (Actual and Predicted - last 30 days)
        for date, actual, predicted in zip(results_df.index[-30:], results_df['Actual'].tail(30), results_df['Predicted'].tail(30)):
            ax1.text(date, actual + 1, f'{actual:.0f}', ha='center', va='bottom', fontsize=8, color='royalblue')
            ax1.text(date, predicted + 1, f'{predicted:.0f}', ha='center', va='bottom', fontsize=8, color='tomato')

        # Add value labels to Future Forecast plot
        for date, forecast in zip(future_df_with_prices.index, future_df_with_prices['Future_Forecast']):
            ax1.text(date, forecast + 1, f'{forecast:.0f}', ha='center', va='bottom', fontsize=8, color='green')


        # Ensure 'our_price' exists before plotting base price
        if 'our_price' in self.daily_df.columns:
            base_price = self.daily_df['our_price'].mean()
            ax2.plot(future_df_with_prices.index, future_df_with_prices['Price_Recommendation'], label='Price Recommendation', color='darkred', marker='o')
            ax2.axhline(y=base_price, color='gray', linestyle='--', label=f'Base Price (${base_price:.2f})')
            ax2.set_ylabel('Recommended Price ($)'); ax2.legend(); ax2.grid(True)
            ax2.set_title('Dynamic Price Recommendations')

            # Add value labels to Price Recommendation plot
            for date, price in zip(future_df_with_prices.index, future_df_with_prices['Price_Recommendation']):
                 ax2.text(date, price + 0.005, f'${price:.2f}', ha='center', va='bottom', fontsize=8, color='darkred')
        else:
            print("Warning: 'our_price' column not available, skipping price recommendation plot.")
            ax2.set_title('Dynamic Price Recommendations (Data Missing)')
            ax2.set_ylabel('Recommended Price ($)'); ax2.grid(True)


        plt.xlabel('Date'); plt.xticks(rotation=45); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    # --- NEW BUSINESS INSIGHTS FUNCTION ---
    def business_insights(self, results_df: pd.DataFrame, future_df_with_prices: pd.DataFrame, mae: float):
        """Generates and prints business insights for the segment."""
        print(f"\n--- Business Insights for Segment: {self.segment} ---")

        # 1. Model Accuracy
        print(f"1. Forecast Accuracy: The model for '{self.segment}' achieved a Mean Absolute Error (MAE) of {mae:.2f} units on the test data.")
        print(f"   - Insight: On average, the model's daily sales forecast for this segment is off by approximately {mae:.0f} units. This provides a measure of forecast reliability for inventory planning.")

        # 2. Future Demand Overview
        total_future_demand = future_df_with_prices['Future_Forecast'].sum()
        avg_future_demand = future_df_with_prices['Future_Forecast'].mean()
        print(f"\n2. Future Demand Outlook: Over the next {self.config['future_forecast_days']} days, a total of {total_future_demand:.0f} units are forecasted for this segment.")
        print(f"   - Insight: This translates to an average daily demand of {avg_future_demand:.2f} units. Use this for overall stock level planning.")

        # 3. Peak Demand Period
        peak_demand_day = future_df_with_prices['Future_Forecast'].idxmax()
        peak_demand_value = future_df_with_prices['Future_Forecast'].max()
        print(f"\n3. Peak Demand: The highest forecasted demand ({peak_demand_value:.0f} units) occurs around {peak_demand_day.strftime('%Y-%m-%d')}.")
        print("   - Insight: Be prepared for a surge in demand around this date. Consider extra staffing or stocking up.")

        # 4. Low Demand Period
        low_demand_day = future_df_with_prices['Future_Forecast'].idxmin()
        low_demand_value = future_df_with_prices['Future_Forecast'].min()
        print(f"\n4. Low Demand: The lowest forecasted demand ({low_demand_value:.0f} units) occurs around {low_demand_day.strftime('%Y-%m-%d')}.")
        print("   - Insight: This period might be suitable for promotions or clearing excess inventory.")

        # 5. Dynamic Pricing Strategy
        avg_rec_price = future_df_with_prices['Price_Recommendation'].mean()
        base_price = self.daily_df['our_price'].mean() if 'our_price' in self.daily_df.columns else np.nan
        print(f"\n5. Dynamic Pricing: The average recommended price for this segment is ${avg_rec_price:.2f}.")
        if base_price is not np.nan:
             print(f"   - Insight: This is {avg_rec_price - base_price:.2f} {'higher' if avg_rec_price > base_price else 'lower'} than the historical average price (${base_price:.2f}). The dynamic pricing adjusts based on forecasted demand.")

        # 6. Price Recommendation Range
        min_rec_price = future_df_with_prices['Price_Recommendation'].min()
        max_rec_price = future_df_with_prices['Price_Recommendation'].max()
        print(f"\n6. Price Variation: Recommended prices for this segment range from ${min_rec_price:.2f} to ${max_rec_price:.2f}.")
        print("   - Insight: The price recommendations adapt to fluctuations in demand. Be ready to implement these price changes.")

        # 7. Potential Revenue/Profit (if cost is available)
        if 'our_price' in self.daily_df.columns and self.config.get('unit_cost') is not None:
             unit_cost = self.config['unit_cost']
             future_revenue = (future_df_with_prices['Future_Forecast'] * future_df_with_prices['Price_Recommendation']).sum()
             future_profit = (future_df_with_prices['Future_Forecast'] * (future_df_with_prices['Price_Recommendation'] - unit_cost)).sum()
             print(f"\n7. Estimated Financial Impact: Based on the forecast and recommended prices, the estimated total revenue is ${future_revenue:.2f} and estimated total profit is ${future_profit:.2f} over the next {self.config['future_forecast_days']} days.")
             print("   - Insight: This provides a potential financial outcome of implementing the dynamic pricing strategy for this segment.")
        else:
             print("\n7. Estimated Financial Impact: Unit cost or 'our_price' data not available to estimate future revenue/profit.")

        # 8. Segment-Specific Behavior (based on general knowledge or other analysis)
        # This is a placeholder - you would add insights specific to what defines this segment
        print(f"\n8. Segment Profile: This segment ('{self.segment}') likely has specific purchasing behaviors (e.g., price sensitivity, purchase frequency) that are captured by the model's forecast pattern.")
        print("   - Insight: Consider combining these forecasts with broader segment understanding for targeted marketing or product development efforts.")

        print(f"\n--- End of Insights for Segment: {self.segment} ---")


    def run(self) -> Optional[Dict]:
        """Executes the entire forecasting and pricing pipeline for the segment."""
        print(f"\n{'='*60}\n--- Starting LSTM Forecast Pipeline for Product: {self.product_code}, Segment: {self.segment} ---\n{'='*60}")

        if not self._prepare_data():
            return None

        train_loader, val_loader, test_loader, y_test = self._build_dataloaders()

        model_params = {'input_size': self.daily_df.shape[1], 'hidden_size': 128, 'num_layers': 2, 'output_size': self.config['forecast_horizon']}
        history = self._train(train_loader, val_loader, model_params)

        results_df, mae = self._evaluate(test_loader, y_test)
        future_df = self._forecast_future()
        future_df_with_prices = self._calculate_dynamic_prices(future_df)

        self._visualize(results_df, future_df_with_prices)
        self.business_insights(results_df, future_df_with_prices, mae) # Call the new insights function


        print(f"\n--- Future Forecast & Price Table for {self.segment} ---")
        print(future_df_with_prices.round(2))
        print(f"\n--- Pipeline Finished for Segment: {self.segment} ---")

        # Return all necessary data for the final summary
        return {
            'segment': self.segment,
            'mae': mae,
            'total_forecast_demand': future_df_with_prices['Future_Forecast'].sum(),
            'avg_recommended_price': future_df_with_prices['Price_Recommendation'].mean(),
            'forecast_df': future_df_with_prices
        }

# --- 3. UTILITY FUNCTIONS ---
def check_file_paths(config: Dict) -> bool:
    """Checks if all required data files exist before starting the pipeline."""
    paths_to_check = ['sales_path', 'competitor_path', 'segments_path']
    for path_key in paths_to_check:
        # Only check if the path is not empty
        if config.get(path_key) and not os.path.exists(config[path_key]):
            print(f"FATAL ERROR: File not found at the specified path: '{config[path_key]}'")
            print(f"Please update the '{path_key}' path in the PIPELINE_CONFIG dictionary.")
            return False
    return True


def generate_overall_business_insights(results_list: List[Dict], product_code: str):
    """Generates a high-level summary of insights across all segments."""
    if not results_list:
        print("\nNo segments were successfully processed. Cannot generate overall insights.")
        return

    summary_df = pd.DataFrame(results_list)

    # Ensure 'mae' is numeric for sorting
    summary_df['mae'] = pd.to_numeric(summary_df['mae'], errors='coerce')
    summary_df.dropna(subset=['mae'], inplace=True) # Drop segments where MAE couldn't be calculated

    if summary_df.empty:
         print("\nNo segments with valid MAE results were processed. Cannot generate overall insights.")
         return

    # Sort by MAE to find most accurate, handle potential NaNs from skipped segments
    most_accurate_segment = summary_df.loc[summary_df['mae'].idxmin()] if not summary_df['mae'].isnull().all() else None


    # Sort by total forecast demand
    summary_df['total_forecast_demand'] = pd.to_numeric(summary_df['total_forecast_demand'], errors='coerce')
    highest_demand_segment = summary_df.loc[summary_df['total_forecast_demand'].idxmax()] if not summary_df['total_forecast_demand'].isnull().all() else None

    # Sort by average recommended price
    summary_df['avg_recommended_price'] = pd.to_numeric(summary_df['avg_recommended_price'], errors='coerce')
    highest_price_segment = summary_df.loc[summary_df['avg_recommended_price'].idxmax()] if not summary_df['avg_recommended_price'].isnull().all() else None

    total_demand = summary_df['total_forecast_demand'].sum() if not summary_df['total_forecast_demand'].isnull().all() else 0


    # --- Text-Based Insights ---
    print("\n" + "="*70)
    print("---           OVERALL BUSINESS INSIGHTS SUMMARY           ---")
    print("="*70)
    print(f"\nAnalysis for Product Code: {product_code}\n")

    if most_accurate_segment is not None:
        print(f"🎯 Most Accurate Forecast: '{most_accurate_segment['segment']}'")
        print(f"   - Model for this segment has the lowest error (MAE: {most_accurate_segment['mae']:.2f}), making its forecast the most reliable.")
    else:
         print("🎯 Most Accurate Forecast: N/A (Could not compute MAE for any segment)")


    if highest_demand_segment is not None:
        print(f"\n📈 Highest Demand Segment: '{highest_demand_segment['segment']}'")
        print(f"   - This group is predicted to buy the most units ({highest_demand_segment['total_forecast_demand']:.0f} units) in the next {PIPELINE_CONFIG.get('future_forecast_days', 15)} days.")
        print(f"   - STRATEGY: Ensure sufficient stock for this segment and consider targeted marketing.")
    else:
        print("\n📈 Highest Demand Segment: N/A (Could not compute total forecast demand for any segment)")


    if highest_price_segment is not None:
        print(f"\n💰 Highest Price Tolerance: '{highest_price_segment['segment']}'")
        print(f"   - This group's demand forecast supports the highest average price (${highest_price_segment['avg_recommended_price']:.2f}).")
        print(f"   - STRATEGY: Focus premium product versions or bundles on this segment.")
    else:
        print("\n💰 Highest Price Tolerance: N/A (Could not compute average recommended price for any segment)")


    print(f"\n📦 Total Forecasted Demand (All Processed Segments):")
    print(f"   - A total of {total_demand:.0f} units are predicted to be sold in the next {PIPELINE_CONFIG.get('future_forecast_days', 15)} days across segments with sufficient data.")

    print(f"\n📊 Insights from Combined Plot:")
    if any('forecast_df' in result and not result['forecast_df'].empty for result in results_list):
         if highest_demand_segment is not None:
              print(f"   - VISUAL ANALYSIS: The combined forecast plot shows that the demand for '{highest_demand_segment['segment']}' is consistently higher than other segments.")
         if highest_price_segment is not None:
              print(f"   - PRICING DYNAMICS: The recommended prices for '{highest_price_segment['segment']}' are visibly higher, confirming their lower price sensitivity.")
    else:
        print("   - Combined plot not generated due to insufficient data in processed segments.")

    print("\n" + "="*70)


# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    try:
        PIPELINE_CONFIG = {
            'sales_path': "/content/refine_file.csv",
            'competitor_path': "/content/competitor_prices.csv",
            'segments_path': "/content/customer_segmentation_kmeans_results.csv",
            'product_code': '85099B',
            'date_col': 'InvoiceDate',
            'date_format': '%d-%m-%Y %H:%M',
            'quantity_col': 'Quantity',
            'customer_id_col': 'Customer ID', # Explicitly define customer ID column
            'seq_length': 60,
            'forecast_horizon': 1,
            'train_split': 0.7,
            'val_split': 0.15,
            'learning_rate': 0.001, # Adjusted learning rate
            'epochs': 100,
            'patience': 25,
            'future_forecast_days': 15,
            'seed': 42,
            'elasticity': -1.5,
            'intensity': 0.1,
            'holiday_country': 'UK', # Added holiday country config
            'unit_cost': 25.0 # Added unit cost config for insights
        }

        if not check_file_paths(PIPELINE_CONFIG):
            # The check_file_paths function already prints an error message
            pass # Do nothing here, let the check function handle the message and return False

        sales_df = pd.read_csv(PIPELINE_CONFIG['sales_path'])
        segments_df = pd.read_csv(PIPELINE_CONFIG['segments_path'])

        merged_df = pd.merge(sales_df, segments_df, on=PIPELINE_CONFIG['customer_id_col'], how='left')
        merged_df['Cluster_Persona'].fillna('Unknown', inplace=True)

        all_segment_results = []

        for segment in merged_df['Cluster_Persona'].unique():
            forecaster = SegmentForecaster(
                df=merged_df,
                segment=segment,
                product_code=PIPELINE_CONFIG['product_code'],
                competitor_path=PIPELINE_CONFIG['competitor_path'],
                config=PIPELINE_CONFIG
            )
            result = forecaster.run()
            if result: # Only append if the pipeline run for the segment was successful
                all_segment_results.append(result)

        # Generate overall insights based on collected results
        generate_overall_business_insights(all_segment_results, PIPELINE_CONFIG['product_code'])

    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
