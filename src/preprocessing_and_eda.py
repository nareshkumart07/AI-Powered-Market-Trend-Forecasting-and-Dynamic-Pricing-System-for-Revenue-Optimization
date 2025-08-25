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
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        return df
    except FileNotFoundError:
        return None

# --- 2. DATA CLEANING & PREPROCESSING ---

def handle_negative_values(df):
    """Removes rows with negative Quantity or zero Price, which are invalid transactions."""
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
    df_featured['InvoiceYearMonth'] = df_featured['InvoiceDate'].dt.strftime('%Y-%m')
    df_featured['InvoiceWeekday'] = df_featured['InvoiceDate'].dt.day_name()
    df_featured['InvoiceHour'] = df_featured['InvoiceDate'].dt.hour
    return df_featured

def preprocess_pipeline(df):
    """Orchestrates the entire data preprocessing workflow."""
    df = handle_negative_values(df)
    df = remove_outliers_iqr(df, ['Quantity', 'Price'])
    df = impute_missing_values(df)
    df = engineer_features(df)
    return df

# --- 3. ANALYSIS & PLOTTING FUNCTIONS ---

def analyze_top_performers(df, top_n=10):
    """Analyzes and returns figures for top-performing products and countries."""
    figs = []
    # Top Revenue Products
    top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(top_n)
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=top_revenue_products.index, x=top_revenue_products.values, palette='viridis', orient='h', ax=ax1)
    ax1.set_title(f'Top {top_n} Products by Revenue', fontsize=16)
    ax1.set_xlabel('Total Revenue', fontsize=12)
    ax1.set_ylabel('Product Description', fontsize=12)
    figs.append(fig1)

    # Top Quantity Products
    top_quantity_products = df.groupby('Description')['Quantity'].sum().nlargest(top_n)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=top_quantity_products.index, x=top_quantity_products.values, palette='viridis', orient='h', ax=ax2)
    ax2.set_title(f'Top {top_n} Products by Quantity Sold', fontsize=16)
    ax2.set_xlabel('Total Quantity Sold', fontsize=12)
    ax2.set_ylabel('Product Description', fontsize=12)
    figs.append(fig2)

    # Top Revenue Countries
    top_revenue_countries = df.groupby('Country')['Revenue'].sum().nlargest(top_n)
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.barplot(y=top_revenue_countries.index, x=top_revenue_countries.values, palette='viridis', orient='h', ax=ax3)
    ax3.set_title(f'Top {top_n} Countries by Revenue', fontsize=16)
    ax3.set_xlabel('Total Revenue', fontsize=12)
    ax3.set_ylabel('Country', fontsize=12)
    figs.append(fig3)
    
    return figs

def analyze_temporal_trends(df):
    """Analyzes and returns figures for monthly, weekly, and hourly trends."""
    figs = []
    # Monthly Revenue
    monthly_revenue = df.groupby('InvoiceYearMonth')['Revenue'].sum()
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    monthly_revenue.plot(kind='line', marker='o', color='b', ax=ax1)
    ax1.set_title('Monthly Revenue Trend', fontsize=16)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Total Revenue', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    figs.append(fig1)

    # Weekday Revenue
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_revenue = df.groupby('InvoiceWeekday')['Revenue'].sum().reindex(weekday_order)
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    weekday_revenue.plot(kind='line', marker='o', color='g', ax=ax2)
    ax2.set_title('Weekday Revenue Trend', fontsize=16)
    ax2.set_xlabel('Day of the Week', fontsize=12)
    ax2.set_ylabel('Total Revenue', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    figs.append(fig2)
    
    return figs

def analyze_price_and_basket(df):
    """Analyzes and returns figures for price and basket size distributions."""
    figs = []
    # Price Distribution
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.histplot(df['Price'], bins=50, kde=True, color='purple', ax=ax1)
    ax1.set_title('Distribution of Unit Prices', fontsize=16)
    ax1.set_xlabel('Price', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    figs.append(fig1)

    # Basket Size Distribution
    basket_sizes = df.groupby('Invoice')['StockCode'].count()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.histplot(basket_sizes, bins=40, color='orange', kde=True, ax=ax2)
    ax2.set_title('Distribution of Basket Sizes', fontsize=16)
    ax2.set_xlabel('Number of Items in Basket', fontsize=12)
    ax2.set_ylabel('Number of Transactions', fontsize=12)
    ax2.set_xlim(0, basket_sizes.quantile(0.99))
    figs.append(fig2)
    
    return figs

def analyze_customer_behavior(df):
    """Analyzes and returns a figure for new vs. returning customer revenue."""
    known_customers_df = df[df['Customer ID'] != 'Unknown']
    customer_invoice_count = known_customers_df.groupby('Customer ID')['Invoice'].nunique()
    returning_customer_ids = customer_invoice_count[customer_invoice_count > 1].index
    returning_revenue = known_customers_df[known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    new_revenue = known_customers_df[~known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    revenue_data = pd.DataFrame({'Customer Type': ['Returning', 'New'], 'Revenue': [returning_revenue, new_revenue]})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=revenue_data, x='Customer Type', y='Revenue', palette='pastel', ax=ax)
    ax.set_title('Revenue from New vs. Returning Customers', fontsize=16)
    ax.set_ylabel('Total Revenue')
    ax.set_xlabel('Customer Type')
    return fig

def plot_geographic_heatmap(df):
    """Returns a Plotly choropleth map figure."""
    country_revenue = df.groupby('Country')['Revenue'].sum().reset_index()
    fig = px.choropleth(country_revenue,
                        locations="Country", locationmode='country names',
                        color="Revenue", hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Geographic Distribution of Revenue')
    return fig

def analyze_market_basket(df, top_n=15):
    """Analyzes and returns a heatmap figure for market basket analysis."""
    top_products = df['Description'].value_counts().nlargest(top_n).index
    df_top = df[df['Description'].isin(top_products)]
    crosstab = pd.crosstab(df_top['Invoice'], df_top['Description'])
    crosstab[crosstab > 0] = 1
    co_occurrence_matrix = crosstab.T.dot(crosstab)
    np.fill_diagonal(co_occurrence_matrix.values, 0)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(co_occurrence_matrix, annot=True, cmap='YlGnBu', fmt='g', ax=ax)
    ax.set_title(f'Co-occurrence Matrix of Top {top_n} Products', fontsize=16)
    ax.set_xlabel('Product', fontsize=12)
    ax.set_ylabel('Product', fontsize=12)
    return fig

# --- 4. REPORTING ---

def print_summary_report(df):
    """Prints a summary of all key business insights and recommendations."""
    # This function is designed for console output, but we can adapt its logic
    # for display in Streamlit.
    
    # Extract insights
    top_revenue_products = df.groupby('Description')['Revenue'].sum().nlargest(1)
    top_revenue_countries = df.groupby('Country')['Revenue'].sum().nlargest(1)
    monthly_revenue = df.groupby('InvoiceYearMonth')['Revenue'].sum()
    weekday_revenue = df.groupby('InvoiceWeekday')['Revenue'].sum()
    hourly_revenue = df.groupby('InvoiceHour')['Revenue'].sum()
    
    known_customers_df = df[df['Customer ID'] != 'Unknown']
    customer_invoice_count = known_customers_df.groupby('Customer ID')['Invoice'].nunique()
    returning_customer_ids = customer_invoice_count[customer_invoice_count > 1].index
    returning_rev = known_customers_df[known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    new_rev = known_customers_df[~known_customers_df['Customer ID'].isin(returning_customer_ids)]['Revenue'].sum()
    total_rev = returning_rev + new_rev
    returning_share = (returning_rev / total_rev) * 100 if total_rev > 0 else 0

    # Create a dictionary of insights to be returned
    insights = {
        "top_product": top_revenue_products.index[0],
        "top_country": top_revenue_countries.index[0],
        "top_month": monthly_revenue.idxmax(),
        "top_day": weekday_revenue.idxmax(),
        "top_hour": hourly_revenue.idxmax(),
        "returning_revenue": returning_rev,
        "new_revenue": new_rev,
        "returning_share": returning_share
    }
    return insights
