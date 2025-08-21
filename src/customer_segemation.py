"""
This script performs a complete RFM (Recency, Frequency, Monetary) analysis
on customer data. The code is structured into modular, reusable functions for
each step of the analysis, from data preparation to visualization.
"""

# --- 0. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify  # A library for creating treemaps
from typing import Optional

# --- 1. DATA PREPARATION ---

def prepare_rfm_data(
    df: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    revenue_col: str,
    date_format: str
) -> pd.DataFrame:
    """
    Prepares the input DataFrame for RFM analysis.

    - Validates that required columns exist.
    - Converts the date column to datetime objects.
    """
    print("Step 1: Preparing data for RFM analysis...")
    local_df = df.copy()
    required_cols = [customer_id_col, invoice_date_col, revenue_col]
    
    # Ensure all required columns are present in the DataFrame
    assert all(col in local_df.columns for col in required_cols), \
        f"Error: DataFrame must contain the columns: {required_cols}"

    # Convert the invoice date column to datetime objects
    local_df[invoice_date_col] = pd.to_datetime(local_df[invoice_date_col], format=date_format)
    print("-> Converted date column to datetime objects.")
    return local_df

# --- 2. RFM METRIC CALCULATION ---

def calculate_rfm_metrics(
    df: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    revenue_col: str
) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary values for each customer."""
    print("Step 2: Calculating Recency, Frequency, and Monetary metrics...")
    
    # The analysis date is set to one day after the most recent transaction
    analysis_date = df[invoice_date_col].max() + pd.Timedelta(days=1)

    # Calculate R, F, M values
    recency = df.groupby(customer_id_col)[invoice_date_col].max().apply(lambda x: (analysis_date - x).days)
    frequency = df.groupby(customer_id_col)[invoice_date_col].count()
    monetary = df.groupby(customer_id_col)[revenue_col].sum()

    # Combine into a single DataFrame
    rfm = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
    print("-> RFM metrics calculated successfully.")
    return rfm

# --- 3. RFM SCORING AND SEGMENTATION ---

def assign_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns scores from 1 to 5 for each RFM metric based on quintiles."""
    print("Step 3: Assigning RFM scores...")
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    print("-> Scores assigned.")
    return rfm_df

def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Segments customers based on their R and F scores."""
    print("Step 4: Segmenting customers...")
    rf_score_str = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str)
    segment_map = {
        r'[1-2][1-2]': 'Hibernating', r'[1-2][3-4]': 'At Risk', r'[1-2]5': "Can't Lose Them",
        r'3[1-2]': 'About to Sleep', r'33': 'Need Attention', r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising', r'51': 'New Customers', r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }
    rfm_df['Segment'] = rf_score_str.replace(segment_map, regex=True)
    print("-> Customers segmented.")
    print("\nCustomer Segment Distribution:")
    print(rfm_df['Segment'].value_counts())
    return rfm_df

def assign_business_actions(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns suggested business actions to each customer segment."""
    print("Step 5: Assigning business actions...")
    business_actions = {
        "Hibernating": "Re-engage with reminder emails, seasonal promotions.",
        "Loyal Customers": "Strengthen relationship with exclusive offers, loyalty rewards.",
        "Champions": "Reward with VIP programs, personalized gifts, and ambassador opportunities.",
        "At Risk": "Launch win-back campaigns, offer special discounts, and request feedback.",
        "Potential Loyalists": "Nurture with targeted recommendations and membership incentives.",
        "About to Sleep": "Send wake-up campaigns, birthday/anniversary offers.",
        "Need Attention": "Use personalized outreach, highlight trending products, and send surveys.",
        "Can't Lose Them": "Offer strong retention incentives and reactivation bundles.",
        "Promising": "Encourage repeat purchases with targeted recommendations and cross-selling.",
        "New Customers": "Deliver a strong onboarding experience and welcome discounts."
    }
    rfm_df["Action"] = rfm_df["Segment"].map(business_actions)
    print("-> Actions assigned.")
    return rfm_df

# --- 4. VISUALIZATION ---

def plot_segment_distribution(rfm_df: pd.DataFrame):
    """Plots a bar chart of the customer segment distribution."""
    plt.figure(figsize=(12, 6))
    sns.countplot(data=rfm_df, x='Segment', order=rfm_df['Segment'].value_counts().index, palette='viridis')
    plt.title('Customer Segment Distribution', fontsize=16)
    plt.xlabel('Segment', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_recency_vs_frequency(rfm_df: pd.DataFrame):
    """Plots a scatter plot of Recency vs. Frequency, colored by segment ."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=rfm_df, x='Recency', y='Frequency', hue='Segment', palette='viridis', s=80, alpha=0.7)
    plt.title('Recency vs. Frequency by Segment', fontsize=16)
    plt.xlabel('Recency (Days since last purchase)', fontsize=12)
    plt.ylabel('Frequency (Total purchases)', fontsize=12)
    plt.legend(title='Customer Segments')
    plt.grid(True)
    plt.show()

def plot_segment_treemap(rfm_df: pd.DataFrame):
    """Plots a treemap visualizing the proportion of each customer segment."""
    segment_counts = rfm_df['Segment'].value_counts()
    plt.figure(figsize=(14, 9))
    squarify.plot(
        sizes=segment_counts.values,
        label=[f'{label}\n({count})' for label, count in segment_counts.items()],
        color=sns.color_palette("viridis", len(segment_counts)),
        alpha=0.8
    )
    plt.title('Treemap of Customer Segments', fontsize=16)
    plt.axis('off')
    plt.show()

# --- 5. MAIN PIPELINE ORCHESTRATOR ---

def rfm_analysis_pipeline(
    df: pd.DataFrame,
    customer_id_col: str = 'Customer ID',
    invoice_date_col: str = 'InvoiceDate',
    revenue_col: str = 'Revenue',
    date_format: str = '%d-%m-%Y %H:%M',
    output_csv_path: Optional[str] = "customer_segments_with_actions.csv",
    generate_plots: bool = True
) -> pd.DataFrame:
    """
    Orchestrates the complete RFM analysis pipeline from data prep to visualization.
    """
    # Run the analysis steps in sequence
    prepared_df = prepare_rfm_data(df, customer_id_col, invoice_date_col, revenue_col, date_format)
    rfm_metrics = calculate_rfm_metrics(prepared_df, customer_id_col, invoice_date_col, revenue_col)
    rfm_scored = assign_rfm_scores(rfm_metrics)
    rfm_segmented = segment_customers(rfm_scored)
    rfm_final = assign_business_actions(rfm_segmented)

    # Save results to a CSV file if a path is provided
    if output_csv_path:
        rfm_final.to_csv(output_csv_path, index=True)
        print(f"\n✅ Analysis complete. Results saved to: {output_csv_path}")

    # Generate and display all visualizations if requested
    if generate_plots:
        print("\nGenerating visualizations...")
        plot_segment_distribution(rfm_final)
        plot_recency_vs_frequency(rfm_final)
        plot_segment_treemap(rfm_final)

    return rfm_final

# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    try:
        # 1. Define the path to your data file
        file_path = '/content/refine_file.csv'
        input_df = pd.read_csv(file_path)

        # 2. Run the RFM analysis by calling the main pipeline function
        rfm_results_df = rfm_analysis_pipeline(
            df=input_df,
            output_csv_path="customer_segmentation_results.csv",
            generate_plots=True
        )

        # 3. Display the first few rows of the final results
        print("\n--- RFM Analysis Output (First 5 Rows) ---")
        print(rfm_results_df.head())

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
