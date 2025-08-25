"""
This script performs a comprehensive Exploratory Data Analysis (EDA) on the
Online Retail II dataset. It is structured into modular functions for loading,
preprocessing, visualizing, and analyzing the data to derive actionable
business insights.
"""

# --- 0. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime as dt

# --- 1. DATA LOADING ---

def load_data(filepath):
    """Loads the dataset from an Excel file."""
    print("Loading data...")
    try:
        # The engine='openpyxl' is important for reading .xlsx files
        df = pd.read_excel(filepath, engine='openpyxl')
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

# --- 2. DATA CLEANING & PREPROCESSING ---

def handle_negative_values(df):
    """Removes rows with negative Quantity or zero Price, which are invalid transactions."""
    print("Handling negative or zero values in 'Quantity' and 'Price'...")
    return df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()

def remove_outliers_iqr(df, columns):
    """Removes outliers from specified columns using the IQR method."""
    print(f"Removing outliers from {columns} using IQR method...")
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter the dataframe to keep values within the calculated bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def impute_missing_values(df):
    """Handles missing values for 'Customer ID' and 'Description'."""
    print("Imputing missing values...")
    df_imputed = df.copy()
    # Fill missing Customer IDs with 'Unknown' for tracking anonymous purchases
    df_imputed['Customer ID'].fillna('Unknown', inplace=True)
    # Create a mapping from StockCode to Description to fill missing descriptions
    stockcode_map = df_imputed.dropna(subset=['Description']).set_index('StockCode')['Description'].to_dict()
    df_imputed['Description'] = df_imputed['Description'].fillna(df_imputed['StockCode'].map(stockcode_map))
    # If any descriptions are still missing, label them as 'Unknown'
    df_imputed['Description'].fillna('Unknown', inplace=True)
    return df_imputed

def engineer_features(df):
    """Creates new features like Revenue and time-based attributes."""
    print("Engineering new features...")
    df_featured = df.copy()
    # Calculate total revenue for each transaction
    df_featured['Revenue'] = df_featured['Quantity'] * df_featured['Price']
    # Convert InvoiceDate to datetime objects for time-series analysis
    df_featured['InvoiceDate'] = pd.to_datetime(df_featured['InvoiceDate'])
    # Extract temporal features for trend analysis
    df_featured['InvoiceMonth'] = df_featured['InvoiceDate'].dt.to_period('M')
    df_featured['InvoiceYearMonth'] = df_featured['InvoiceDate'].dt.strftime('%Y-%m')
    df_featured['InvoiceWeekday'] = df_featured['InvoiceDate'].dt.day_name()
    df_featured['InvoiceHour'] = df_featured['InvoiceDate'].dt.hour
    return df_featured

def preprocess_pipeline(df):
    """Orchestrates the entire data preprocessing workflow."""
    print("\n--- Starting Data Preprocessing Pipeline ---")
    df = handle_negative_values(df)
    print("Visualizing data distribution before outlier removal...")
    visualize_outlier_effect(df, ['Quantity', 'Price'], 'Before')
    df = remove_outliers_iqr(df, ['Quantity', 'Price'])
    print("Visualizing data distribution after outlier removal...")
    visualize_outlier_effect(df, ['Quantity', 'Price'], 'After')
    df = impute_missing_values(df)
    df = engineer_features(df)
    print("--- Preprocessing Complete ---\n")
    return df

# --- 3. VISUALIZATION FUNCTIONS ---

def visualize_outlier_effect(df, columns, stage):
    """Visualizes the distribution of specified columns using boxplots."""
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.boxplot(x=df[col])
        plt.title(f'{stage} Outlier Removal: Distribution of {col}', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_top_n(data, title, xlabel, ylabel):
    """Generic function to plot top N horizontal bar charts."""
    plt.figure(figsize=(12, 8))
    sns.barplot(y=data.index, x=data.values, palette='viridis', orient='h')
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_temporal_trend(data, title, xlabel, ylabel, color, xticks_range=None):
    """Generic function to plot line charts for temporal trends."""
    plt.figure(figsize=(14, 7))
    data.plot(kind='line', marker='o', color=color, legend=False)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if xticks_range:
        plt.xticks(ticks=range(xticks_range[0], xticks_range[1] + 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_geographic_heatmap(df):
    """Plots a choropleth map of revenue by country using Plotly."""
    print("\nAnalyzing geographic distribution of revenue...")
    country_revenue = df.groupby('Country')['Revenue'].sum().reset_index()
    fig = px.choropleth(country_revenue,
                        locations="Country",
                        locationmode='country names',
                        color="Revenue",
                        hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Geographic Distribution of Revenue')
    fig.show()

# --- 4. ANALYSIS FUNCTIONS ---

# ## 4.1. Descriptive Analytics ##
def analyze_top_performers(df, top_n=10):
    """Analyzes top-performing products and countries by revenue and quantity."""
    print("\n--- Analyzing Top Performers ---")
    top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(top_n)
    top_quantity_products = df.groupby('Description')['Quantity'].sum().nlargest(top_n)
    top_revenue_countries = df.groupby('Country')['Revenue'].sum().nlargest(top_n)

    plot_top_n(top_revenue_products, f'Top {top_n} Products by Revenue', 'Total Revenue', 'Product Description')
    plot_top_n(top_quantity_products, f'Top {top_n} Products by Quantity Sold', 'Total Quantity Sold', 'Product Description')
    plot_top_n(top_revenue_countries, f'Top {top_n} Countries by Revenue', 'Total Revenue', 'Country')

    return {'top_revenue_products': top_revenue_products, 'top_revenue_countries': top_revenue_countries}

def analyze_temporal_trends(df):
    """Analyzes monthly, weekly, and hourly revenue trends."""
    print("\n--- Analyzing Temporal Trends ---")
    monthly_revenue = df.groupby('InvoiceYearMonth')['Revenue'].sum()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_revenue = df.groupby('InvoiceWeekday')['Revenue'].sum().reindex(weekday_order)
    hourly_revenue = df.groupby('InvoiceHour')['Revenue'].sum()

    plot_temporal_trend(monthly_revenue, 'Monthly Revenue Trend', 'Month', 'Total Revenue', 'b')
    plot_temporal_trend(weekday_revenue, 'Weekday Revenue Trend', 'Day of the Week', 'Total Revenue', 'g')
    plot_temporal_trend(hourly_revenue, 'Hourly Revenue Trend', 'Hour of the Day', 'Total Revenue', 'r', (0, 23))

    return {'monthly': monthly_revenue, 'weekday': weekday_revenue, 'hourly': hourly_revenue}

def analyze_price_and_basket(df):
    """Analyzes price distribution and transaction basket size."""
    print("\n--- Analyzing Price and Basket Size ---")
    # Price Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Price'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Unit Prices', fontsize=16)
    plt.xlabel('Price', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

    # Basket Size Distribution
    basket_sizes = df.groupby('Invoice')['StockCode'].count()
    plt.figure(figsize=(12, 6))
    sns.histplot(basket_sizes, bins=40, color='orange', kde=True)
    plt.title('Distribution of Basket Sizes (Items per Transaction)', fontsize=16)
    plt.xlabel('Number of Items in Basket', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    # Limit x-axis to the 99th percentile for better readability
    plt.xlim(0, basket_sizes.quantile(0.99))
    plt.show()

# ## 4.2. Customer Behavior ##
def analyze_customer_behavior(df):
    """Analyzes new vs. returning customer revenue."""
    print("\n--- Analyzing Customer Behavior (New vs. Returning) ---")
    known_customers_df = df[df['Customer ID'] != 'Unknown']
    customer_invoice_count = known_customers_df.groupby('Customer ID')['Invoice'].nunique()
    returning_customer_ids = customer_invoice_count[customer_invoice_count > 1].index
    returning_revenue = known_customers_df[known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    new_revenue = known_customers_df[~known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    revenue_data = pd.DataFrame({'Customer Type': ['Returning', 'New'], 'Revenue': [returning_revenue, new_revenue]})
    plt.figure(figsize=(8, 6))
    sns.barplot(data=revenue_data, x='Customer Type', y='Revenue', palette='pastel')
    plt.title('Revenue from New vs. Returning Customers', fontsize=16)
    plt.ylabel('Total Revenue'); plt.xlabel('Customer Type')
    plt.show()
    return {'returning_revenue': returning_revenue, 'new_revenue': new_revenue}


# ## 4.3. Product Affinity ##
def analyze_market_basket(df, top_n=15):
    """Performs a simple market basket analysis to find co-occurring products."""
    print("\n--- Performing Market Basket Analysis ---")
    # Focus on top N products for a cleaner visualization
    top_products = df['Description'].value_counts().nlargest(top_n).index
    df_top = df[df['Description'].isin(top_products)]

    # Create an invoice-product matrix (crosstab)
    crosstab = pd.crosstab(df_top['Invoice'], df_top['Description'])
    # Convert counts to binary (present or not)
    crosstab[crosstab > 0] = 1

    # Calculate co-occurrence matrix
    co_occurrence_matrix = crosstab.T.dot(crosstab)
    # Set diagonal to zero to remove self-pairing
    np.fill_diagonal(co_occurrence_matrix.values, 0)

    # Plot the heatmap
    plt.figure(figsize=(12, 12))
    sns.heatmap(co_occurrence_matrix, annot=True, cmap='YlGnBu', fmt='g')
    plt.title(f'Co-occurrence Matrix of Top {top_n} Products', fontsize=16)
    plt.xlabel('Product', fontsize=12)
    plt.ylabel('Product', fontsize=12)
    plt.show()

# --- 5. REPORTING ---

def print_summary_report(performer_data, temporal_data, customer_data):
    """Prints a summary of all key business insights and recommendations."""
    # Extract insights
    top_product = performer_data['top_revenue_products'].index[0]
    top_country = performer_data['top_revenue_countries'].index[0]
    top_month = temporal_data['monthly'].idxmax()
    top_day = temporal_data['weekday'].idxmax()
    top_hour = temporal_data['hourly'].idxmax()
    
    returning_rev = customer_data['returning_revenue']
    new_rev = customer_data['new_revenue']
    total_rev = returning_rev + new_rev
    returning_share = (returning_rev / total_rev) * 100 if total_rev > 0 else 0

    print("\n" + "="*70)
    print("           ACTIONABLE BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("="*70 + "\n")
    print("🚀 Top Performers & Peak Times:")
    print(f"  - Top Revenue Product: '{top_product}'")
    print(f"  - Top Market (Country): '{top_country}'")
    print(f"  - Peak Sales Month: {top_month}")
    print(f"  - Busiest Day of the Week: {top_day}")
    print(f"  - Peak Sales Hour: {top_hour}:00 - {top_hour+1}:00")
    print("\n" + "-"*70 + "\n")
    print("💡 Customer Behavior Insights:")
    print(f"  - Revenue from Returning Customers: ${returning_rev:,.2f}")
    print(f"  - Revenue from New Customers: ${new_rev:,.2f}")
    print(f"  - Returning customers drive {returning_share:.1f}% of identified customer revenue.")
    print("\n" + "="*70)
    print("\n🎯 Strategic Recommendations:")
    print("  1. Marketing & Sales:")
    print(f"     - Launch targeted email campaigns on your busiest day/hour ({top_day} around {top_hour}:00).")
    print(f"     - Increase marketing spend and targeted ads in your top market: '{top_country}'.")
    print("  2. Inventory Management:")
    print(f"     - Ensure the top revenue product, '{top_product}', is always well-stocked, especially before the peak month ({top_month}).")
    print("     - Use market basket insights to create product bundles or 'frequently bought together' recommendations on your website.")
    print("  3. Customer Retention:")
    print("     - Since returning customers are vital, invest in loyalty programs, personalized offers, and excellent customer service to maintain their engagement.")
    print("     - Encourage new customers to make a second purchase through a welcome email series with a small discount on their next order.")
    print("="*70)


# --- 6. MAIN EXECUTION ---

def main():
    """Main function to run the entire analysis pipeline."""
    # IMPORTANT: Update this filepath to the location of your data file.
    filepath = '/content/retail_sample_2000.xlsx'

    # Step 1: Load and Preprocess Data
    raw_df = load_data(filepath)
    if raw_df is None:
        return
    clean_df = preprocess_pipeline(raw_df)

    # Step 2: Perform Core EDA
    performer_data = analyze_top_performers(clean_df)
    temporal_data = analyze_temporal_trends(clean_df)
    customer_data = analyze_customer_behavior(clean_df)
    plot_geographic_heatmap(clean_df)
    analyze_price_and_basket(clean_df)

    # Step 3: Perform Product Affinity Analysis
    analyze_market_basket(clean_df)
    
    # Step 4: Print the final summary report
    print_summary_report(performer_data, temporal_data, customer_data)

if __name__ == "__main__":
    main()
