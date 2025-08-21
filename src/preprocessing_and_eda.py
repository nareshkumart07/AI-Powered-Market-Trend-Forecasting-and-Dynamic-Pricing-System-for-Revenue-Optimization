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

# --- 1. DATA LOADING ---

def load_data(filepath):
    """Loads the dataset from an Excel file."""
    print("Loading data...")
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

# --- 2. DATA CLEANING & PREPROCESSING ---

def handle_negative_values(df):
    """Removes rows with negative Quantity or Price."""
    return df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()

def remove_outliers_iqr(df, columns):
    """Removes outliers from specified columns using the IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def impute_missing_values(df):
    """Handles missing values for 'Customer ID' and 'Description'."""
    df_imputed = df.copy()
    df_imputed['Customer ID'].fillna('Unknown', inplace=True)
    stockcode_map = df_imputed.dropna(subset=['Description']).set_index('StockCode')['Description'].to_dict()
    df_imputed['Description'] = df_imputed['Description'].fillna(df_imputed['StockCode'].map(stockcode_map))
    df_imputed['Description'].fillna('Unknown', inplace=True)
    return df_imputed

def engineer_features(df):
    """Creates new features like Revenue and time-based attributes."""
    df_featured = df.copy()
    df_featured['Revenue'] = df_featured['Quantity'] * df_featured['Price']
    df_featured['InvoiceDate'] = pd.to_datetime(df_featured['InvoiceDate'])
    df_featured['InvoiceMonth'] = df_featured['InvoiceDate'].dt.month
    df_featured['InvoiceWeekday'] = df_featured['InvoiceDate'].dt.day_name()
    df_featured['InvoiceHour'] = df_featured['InvoiceDate'].dt.hour
    return df_featured

def preprocess_pipeline(df):
    """Orchestrates the entire data preprocessing workflow."""
    print("Starting data preprocessing pipeline...")
    df = handle_negative_values(df)
    print("Visualizing data distribution before outlier removal...")
    visualize_outlier_effect(df, ['Quantity', 'Price'], 'Before')
    df = remove_outliers_iqr(df, ['Quantity', 'Price'])
    print("Visualizing data distribution after outlier removal...")
    visualize_outlier_effect(df, ['Quantity', 'Price'], 'After')
    df = impute_missing_values(df)
    df = engineer_features(df)
    print("Preprocessing complete.")
    return df

# --- 3. VISUALIZATION FUNCTIONS ---

def visualize_outlier_effect(df, columns, stage):
    """Visualizes the distribution of specified columns using boxplots."""
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, len(columns), i)
        sns.boxplot(x=df[col])
        plt.title(f'{stage} Outlier Removal: {col}', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_top_n(data, title, xlabel, ylabel):
    """Generic function to plot top N bar charts."""
    plt.figure(figsize=(12, 7))
    sns.barplot(y=data.index, x=data.values, palette='viridis', orient='h')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_temporal_trend(data, title, xlabel, ylabel, color, xticks_range):
    """Generic function to plot line charts for temporal trends."""
    plt.figure(figsize=(12, 6))
    data.plot(kind='line', marker='o', color=color)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if xticks_range:
        plt.xticks(ticks=range(xticks_range[0], xticks_range[1]))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_geographic_heatmap(df):
    """Plots a choropleth map of revenue by country."""
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

# ## Original Analysis Functions ##
def analyze_top_performers_by_revenue(df, top_n=10):
    """Analyzes top-performing products and countries by revenue."""
    print("\nAnalyzing top performers by REVENUE...")
    top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(top_n)
    top_revenue_countries = df.groupby('Country')['Revenue'].sum().nlargest(top_n)
    plot_top_n(top_revenue_products, f'Top {top_n} Products by Revenue', 'Total Revenue', 'Product Description')
    return {'top_products': top_revenue_products, 'top_countries': top_revenue_countries}

def analyze_temporal_trends(df):
    """Analyzes monthly, weekly, and hourly revenue trends."""
    print("\nAnalyzing temporal trends...")
    monthly_revenue = df.groupby('InvoiceMonth')['Revenue'].sum()
    weekday_revenue = df.groupby('InvoiceWeekday')['Revenue'].sum()
    hourly_revenue = df.groupby('InvoiceHour')['Revenue'].sum()
    plot_temporal_trend(monthly_revenue, 'Monthly Revenue Trend', 'Month', 'Total Revenue', 'b', (1, 13))
    plot_temporal_trend(hourly_revenue, 'Hourly Revenue Trend', 'Hour of the Day', 'Total Revenue', 'r', (0, 24))
    return {'monthly': monthly_revenue, 'weekday': weekday_revenue, 'hourly': hourly_revenue}

def analyze_customer_behavior(df):
    """Analyzes new vs. returning customer revenue."""
    print("\nAnalyzing customer behavior (New vs. Returning)...")
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

# ## --- NEW VISUALS FOR DEEPER BUSINESS INSIGHTS --- ##

def analyze_top_performers_by_quantity(df, top_n=10):
    """Analyzes top-selling products by quantity sold."""
    print("\nAnalyzing top performers by QUANTITY SOLD...")
    top_quantity_products = df.groupby('Description')['Quantity'].sum().nlargest(top_n)
    plot_top_n(top_quantity_products, f'Top {top_n} Products by Quantity Sold', 'Total Quantity Sold', 'Product Description')
    return {'top_quantity_products': top_quantity_products}

def analyze_price_distribution(df):
    """Visualizes the distribution of unit prices."""
    print("\nAnalyzing product price distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], bins=50, kde=True, color='purple')
    plt.title('Distribution of Unit Prices', fontsize=16)
    plt.xlabel('Price', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

def analyze_basket_size(df):
    """Analyzes the number of items per invoice (basket size)."""
    print("\nAnalyzing transaction basket size...")
    basket_sizes = df.groupby('Invoice')['StockCode'].count()
    plt.figure(figsize=(10, 6))
    sns.histplot(basket_sizes, bins=40, color='orange', kde=True)
    plt.title('Distribution of Basket Sizes (Items per Transaction)', fontsize=16)
    plt.xlabel('Number of Items in Basket', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.xlim(0, basket_sizes.quantile(0.99)) # Limit x-axis to 99th percentile for better readability
    plt.show()

def analyze_customer_value_distribution(df):
    """Visualizes the distribution of total revenue per customer."""
    print("\nAnalyzing customer value distribution...")
    customer_revenue = df[df['Customer ID'] != 'Unknown'].groupby('Customer ID')['Revenue'].sum()
    plt.figure(figsize=(12, 7))
    sns.boxenplot(x=customer_revenue, palette='coolwarm')
    plt.title('Distribution of Total Spend per Customer', fontsize=16)
    plt.xlabel('Total Revenue per Customer', fontsize=12)
    plt.xscale('log') # Use a log scale due to high skew
    plt.show()

def plot_correlation_heatmap(df):
    """Plots a correlation heatmap of numerical features."""
    print("\nPlotting feature correlation heatmap...")
    numeric_cols = df[['Quantity', 'Price', 'Revenue', 'InvoiceMonth', 'InvoiceHour']].copy()
    corr_matrix = numeric_cols.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Key Numerical Features', fontsize=16)
    plt.show()


# --- 5. REPORTING ---

def print_summary_report(performer_data, temporal_data, customer_data):
    """Prints a summary of all key business insights and recommendations."""
    # Extract insights
    top_product = performer_data['top_products'].index[0]
    top_country = performer_data['top_countries'].index[0]
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    top_month_name = month_names[temporal_data['monthly'].idxmax()]
    top_day = temporal_data['weekday'].idxmax()
    top_hour = temporal_data['hourly'].idxmax()
    returning_rev = customer_data['returning_revenue']
    new_rev = customer_data['new_revenue']
    total_rev = returning_rev + new_rev
    returning_share = (returning_rev / total_rev) * 100 if total_rev > 0 else 0

    print("\n" + "="*60)
    print("       ACTIONABLE BUSINESS INSIGHTS SUMMARY")
    print("="*60 + "\n")
    print("🚀 Top Performers:")
    print(f"  - Top Revenue Product: '{top_product}'")
    print(f"  - Top Market (Country): '{top_country}'")
    print(f"  - Peak Sales Month: {top_month_name}")
    print(f"  - Busiest Day of the Week: {top_day}")
    print(f"  - Peak Sales Hour: {top_hour}:00 - {top_hour+1}:00")
    print("\n" + "-"*60 + "\n")
    print("📈 Customer Behavior Insights:")
    print(f"  - Revenue from Returning Customers: ${returning_rev:,.2f}")
    print(f"  - Revenue from New Customers: ${new_rev:,.2f}")
    print(f"  - Returning customers drive {returning_share:.1f}% of identified customer revenue.")
    print("\n" + "="*60)
    print("\n💡 Recommendations:")
    print("  1. Marketing: Focus campaigns around peak times (e.g., midday on Thursdays in November). Launch targeted ads in top-performing countries.")
    print("  2. Inventory: Ensure top products (by both revenue and quantity) are well-stocked. Use top quantity sellers as potential items for promotions.")
    print("  3. Customer Retention: Since returning customers are vital, invest heavily in loyalty programs and personalized offers.")
    print("  4. Upselling: Given the typical basket size, create promotions to encourage adding one more item (e.g., 'spend X, get a discount').")
    print("="*60)

# --- 6. MAIN EXECUTION ---

def main():
    """Main function to run the entire analysis pipeline."""
    filepath = '/content/online_retail_II.xlsx'

    # Step 1: Load and Preprocess Data
    raw_df = load_data(filepath)
    if raw_df is None:
        return
    clean_df = preprocess_pipeline(raw_df)

    # Step 2: Perform Core EDA and gather primary insights
    performer_data = analyze_top_performers_by_revenue(clean_df)
    temporal_data = analyze_temporal_trends(clean_df)
    customer_data = analyze_customer_behavior(clean_df)
    plot_geographic_heatmap(clean_df)

    # Step 3: Perform Additional Deep-Dive EDA for more insights
    analyze_top_performers_by_quantity(clean_df)
    analyze_price_distribution(clean_df)
    analyze_basket_size(clean_df)
    analyze_customer_value_distribution(clean_df)
    plot_correlation_heatmap(clean_df)

    # Step 4: Print the final summary report
    print_summary_report(performer_data, temporal_data, customer_data)

if __name__ == "__main__":
    main()
