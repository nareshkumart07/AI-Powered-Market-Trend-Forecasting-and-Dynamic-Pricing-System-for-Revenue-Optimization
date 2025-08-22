import pandas as pd
import numpy as np

# Load the user's sales data to get the date and price range
try:
    sales_df = pd.read_csv("refine_file.csv")
    print("Successfully loaded 'refine_file.csv'.")
except FileNotFoundError:
    print("Error: 'refine_file.csv' not found. Please make sure the file is uploaded.")
    exit()

# --- 1. Determine Date and Price Range ---
sales_df['InvoiceDate'] = pd.to_datetime(sales_df['InvoiceDate'], format='%d-%m-%Y %H:%M', errors='coerce')
sales_df.dropna(subset=['InvoiceDate'], inplace=True)

min_date = sales_df['InvoiceDate'].min().date()
max_date = sales_df['InvoiceDate'].max().date()

# Calculate a realistic base price from the sales data
product_price_series = sales_df[sales_df['StockCode'] == '85099B']['Price']
if product_price_series.empty or product_price_series.mean() == 0:
    base_price = sales_df['Price'][(sales_df['Price'] > 0) & (sales_df['Price'] < 50)].mean()
    price_std = sales_df['Price'][(sales_df['Price'] > 0) & (sales_df['Price'] < 50)].std()
else:
    base_price = product_price_series.mean()
    price_std = product_price_series.std()

print(f"Date range for competitor data: {min_date} to {max_date}")
print(f"Base price for generating competitor data: ${base_price:.2f}")

# --- 2. Generate Competitor and Our Price Data ---
date_range = pd.to_datetime(pd.date_range(start=min_date, end=max_date, freq='D'))
num_days = len(date_range)

# Create a base price trend with seasonality and random noise
np.random.seed(42)
days = np.arange(num_days)
seasonal_effect = np.sin(2 * np.pi * days / 365) * (base_price * 0.1)
random_noise = np.random.normal(0, price_std * 0.5, size=num_days)
base_trend = base_price + seasonal_effect + random_noise

# Generate prices for three distinct competitors and our own price
competitor_a_prices = base_trend * np.random.uniform(0.95, 0.99, size=num_days)
competitor_b_prices = base_trend * np.random.uniform(0.98, 1.02, size=num_days)
competitor_c_prices = base_trend * np.random.uniform(1.01, 1.05, size=num_days)
our_prices = base_trend * np.random.uniform(0.99, 1.01, size=num_days) # Our price fluctuates closely around the base

# --- 3. Create, Format, and Save the DataFrame ---
# Create the DataFrame in the specified wide format
price_df = pd.DataFrame({
    'Date': date_range,
    'our_price': our_prices,
    'competitor_A': competitor_a_prices,
    'competitor_B': competitor_b_prices,
    'competitor_C': competitor_c_prices
})

# Ensure prices are not negative
price_columns = ['our_price', 'competitor_A', 'competitor_B', 'competitor_C']
price_df[price_columns] = price_df[price_columns].clip(lower=0.01)

# Round all prices to 2 decimal places
price_df[price_columns] = price_df[price_columns].round(2)

# Save the generated data to a new CSV file
output_filename = "our_and_competitor_prices.csv"
price_df.to_csv(output_filename, index=False, date_format='%Y-%m-%d')

print(f"\nSuccessfully generated our and competitor price data, rounded to 2 decimal places.")
print(f"File saved as '{output_filename}'.")
print("\nHere is a preview of the new data:")
print(price_df.head())
