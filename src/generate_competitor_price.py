import pandas as pd
import numpy as np

# --- 1. Load Your Existing Sales Data ---
try:
    # We load your sales data to get the list of products and their base prices.
    df = pd.read_csv("refine_file.csv")
    print("Successfully loaded 'refine_file.csv'.")
except FileNotFoundError:
    print("\nERROR: 'refine_file.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
    # Exit if the file doesn't exist, as we can't proceed.
    exit()

# --- 2. Identify Unique Products and Their Average Prices ---
# We only need one entry per product to serve as a base for competitor pricing.
# We'll group by 'StockCode' and calculate the mean price for each.
product_prices = df.groupby('StockCode')['Price'].mean().reset_index()
print(f"Found {len(product_prices)} unique products.")


# --- 3. Simulate Competitor Prices ---
# Now, we'll create new columns for our fictional competitors.
# The logic here is to simulate a competitive market where prices are similar
# but not identical.

# Competitor A: Tends to be slightly cheaper (the budget option).
# We'll generate their price as a random value between 8% lower and 2% higher than your price.
product_prices['Competitor_A_Price'] = product_prices['Price'].apply(
    lambda p: p * np.random.uniform(0.92, 1.02)
)

# Competitor B: Tends to be similarly priced (the direct competitor).
# We'll generate their price as a random value between 1% lower and 9% higher than your price.
product_prices['Competitor_B_Price'] = product_prices['Price'].apply(
    lambda p: p * np.random.uniform(0.99, 1.09)
)

# Competitor C: Tends to be slightly more expensive (the premium option).
# We'll generate their price as a random value between 5% and 15% higher than your price.
product_prices['Competitor_C_Price'] = product_prices['Price'].apply(
    lambda p: p * np.random.uniform(1.05, 1.15)
)


# --- 4. Clean and Format the Data ---
# Let's round the prices to two decimal places for a clean, realistic look.
product_prices['Price'] = product_prices['Price'].round(2)
product_prices['Competitor_A_Price'] = product_prices['Competitor_A_Price'].round(2)
product_prices['Competitor_B_Price'] = product_prices['Competitor_B_Price'].round(2)
product_prices['Competitor_C_Price'] = product_prices['Competitor_C_Price'].round(2)


# We'll rename the original 'Price' column to make it clear it's your price.
product_prices.rename(columns={'Price': 'Our_Price'}, inplace=True)


# --- 5. Save to a New CSV File ---
# Finally, we save the new dataframe to a CSV file.
# This file can now be loaded into your main dynamic pricing model.
output_filename = 'competitor_prices.csv'
product_prices.to_csv(output_filename, index=False)

print(f"\nSuccessfully generated competitor prices for 3 competitors!")
print(f"Data saved to '{output_filename}'.")

# --- Display a sample of the generated data ---
print("\n--- Sample of the generated competitor price data ---")
print(product_prices.head())

