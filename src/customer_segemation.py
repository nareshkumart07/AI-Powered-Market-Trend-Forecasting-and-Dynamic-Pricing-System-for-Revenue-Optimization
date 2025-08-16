import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify # A library for creating treemaps
from typing import Optional

def perform_rfm_analysis(
    df: pd.DataFrame,
    customer_id_col: str = 'Customer ID',
    invoice_date_col: str = 'InvoiceDate',
    revenue_col: str = 'Revenue',
    date_format: str = '%d-%m-%Y %H:%M',
    output_csv_path: Optional[str] = "customer_segments_with_actions.csv",
    generate_plots: bool = True
) -> pd.DataFrame:
    """
    Performs a complete RFM (Recency, Frequency, Monetary) analysis on customer data.

    This function takes a DataFrame of transactions, calculates RFM metrics,
    assigns scores and segments to each customer, suggests business actions,
    and optionally saves the results and generates visualizations.

    Args:
        df (pd.DataFrame): The input DataFrame containing transaction data.
        customer_id_col (str): The name of the column for customer identification.
        invoice_date_col (str): The name of the column for the invoice/transaction date.
        revenue_col (str): The name of the column for transaction revenue/value.
        date_format (str): The format of the date string in the invoice_date_col.
        output_csv_path (Optional[str]): Path to save the resulting CSV file.
                                         If None, the file is not saved.
        generate_plots (bool): If True, generates and displays summary plots.

    Returns:
        pd.DataFrame: A DataFrame with RFM scores, segments, and
                      suggested actions for each customer.
    """
    # --- 1. Data Preparation ---
    print("Starting RFM analysis...")
    local_df = df.copy()

    # Ensure required columns exist. If not, this will raise an error.
    required_cols = [customer_id_col, invoice_date_col, revenue_col]
    assert all(col in local_df.columns for col in required_cols), \
        f"Error: DataFrame must contain the columns: {required_cols}"

    # Convert 'InvoiceDate' to datetime objects
    local_df[invoice_date_col] = pd.to_datetime(local_df[invoice_date_col], format=date_format)
    print("Converted date column to datetime objects.")

    # --- 2. Calculate RFM Metrics ---
    # Define the analysis date as one day after the last transaction
    analysis_date = local_df[invoice_date_col].max() + pd.Timedelta(days=1)

    # Calculate Recency, Frequency, and Monetary values for each customer
    recency = local_df.groupby(customer_id_col)[invoice_date_col].max().apply(lambda x: (analysis_date - x).days)
    frequency = local_df.groupby(customer_id_col)[invoice_date_col].count()
    monetary = local_df.groupby(customer_id_col)[revenue_col].sum()

    # Combine into a single RFM DataFrame
    rfm = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
    print("Calculated Recency, Frequency, and Monetary metrics.")

    # --- 3. RFM Scoring ---
    # Assign scores from 1 to 5 for each metric based on quintiles
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    print("Assigned RFM scores.")

    # --- 4. Customer Segmentation ---
    # Combine R and F scores for segmentation
    rf_score_str = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)

    # Define segment mapping
    segment_map = {
        r'[1-2][1-2]': 'Hibernating',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': "Can't Lose Them",
        r'3[1-2]': 'About to Sleep',
        r'33': 'Need Attention',
        r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising',
        r'51': 'New Customers',
        r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }
    rfm['Segment'] = rf_score_str.replace(segment_map, regex=True)
    print("Segmented customers based on RFM scores.")
    print("\nCustomer Segment Distribution:")
    print(rfm['Segment'].value_counts())

    # --- 5. Assign Business Actions ---
    business_actions = {
        "Hibernating": "Re-engage with reminder emails, seasonal promotions, or limited-time discounts.",
        "Loyal Customers": "Strengthen relationship with exclusive offers, loyalty rewards, and early access to new products.",
        "Champions": "Reward with VIP programs, personalized gifts, and brand ambassador opportunities.",
        "At Risk": "Launch win-back campaigns, offer special discounts, and request feedback to understand disengagement.",
        "Potential Loyalists": "Nurture with targeted recommendations, onboarding offers, and membership incentives.",
        "About to Sleep": "Send wake-up campaigns, birthday/anniversary offers, or personalized coupons.",
        "Need Attention": "Use personalized outreach, highlight trending products, and send surveys to keep them active.",
        "Can't Lose Them": "Offer strong retention incentives, special appreciation messages, and reactivation bundles.",
        "Promising": "Encourage repeat purchases with targeted recommendations, cross-selling, and small perks.",
        "New Customers": "Deliver a strong onboarding experience, welcome discounts, and introduce best-sellers."
    }
    rfm["Action"] = rfm["Segment"].map(business_actions)
    print("Assigned business actions to each segment.")

    # --- 6. Save and Visualize ---
    if output_csv_path:
        rfm.to_csv(output_csv_path, index=True)
        print(f"\n✅ Analysis complete. Results saved to: {output_csv_path}")

    if generate_plots:
        print("Generating visualizations...")
        # Bar Chart for Segment Distribution
        plt.figure(figsize=(12, 6))
        sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index, palette='viridis')
        plt.title('Customer Segment Distribution')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Scatter Plot for Recency vs. Frequency
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='Segment', palette='viridis', s=80, alpha=0.7)
        plt.title('Recency vs. Frequency by Segment')
        plt.xlabel('Recency (Days since last purchase)')
        plt.ylabel('Frequency (Total purchases)')
        plt.legend(title='Customer Segments')
        plt.grid(True)
        plt.show()

        # Treemap for Segment Proportions
        segment_counts = rfm['Segment'].value_counts()
        plt.figure(figsize=(14, 9))
        squarify.plot(sizes=segment_counts.values,
                      label=[f'{label}\n({count})' for label, count in segment_counts.items()],
                      color=sns.color_palette("viridis", len(segment_counts)),
                      alpha=0.8)
        plt.title('Treemap of Customer Segments')
        plt.axis('off')
        plt.show()

    return rfm

# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    # 1. Load your data
    # If the file doesn't exist, Python will raise a FileNotFoundError.
    file_path = '/content/drive/MyDrive/refine_file.csv'
    input_df = pd.read_csv(file_path)

    # 2. Run the RFM analysis by calling the function
    rfm_results_df = perform_rfm_analysis(
        df=input_df,
        output_csv_path="customer_segmentation_results.csv",
        generate_plots=True
    )

    # 3. Use the results for the next step in your pipeline
    print("\n--- RFM Analysis Output (First 5 Rows) ---")
    print(rfm_results_df.head())
