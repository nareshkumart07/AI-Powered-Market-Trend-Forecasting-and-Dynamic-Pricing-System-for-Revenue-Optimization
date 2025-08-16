import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_rfm_analysis(
    df: pd.DataFrame,
    customer_id_col: str = 'Customer ID',
    invoice_date_col: str = 'InvoiceDate',
    revenue_col: str = 'Revenue',
    date_format: str = '%d-%m-%Y %H:%M',
    output_csv_path: Optional[str] = "customer_segments_with_actions.csv",
    generate_plots: bool = True,
    perform_kmeans: bool = True, # Defaulting to True to focus on K-Means
    n_clusters: int = 4
) -> pd.DataFrame:
    """
    Performs RFM analysis and K-Means clustering to segment customers.

    This function calculates RFM metrics and then uses K-Means clustering
    to group customers into data-driven segments. It analyzes each cluster
    to provide clear business insights.

    Args:
        df (pd.DataFrame): The input DataFrame with transaction data.
        customer_id_col (str): Column name for customer ID.
        invoice_date_col (str): Column name for invoice date.
        revenue_col (str): Column name for transaction revenue.
        date_format (str): The format of the date string.
        output_csv_path (Optional[str]): Path to save the results CSV. If None, not saved.
        generate_plots (bool): If True, generates and displays summary plots.
        perform_kmeans (bool): If True, performs K-Means clustering on RFM metrics.
        n_clusters (int): The number of clusters to form for K-Means.

    Returns:
        pd.DataFrame: A DataFrame with RFM values and K-Means cluster labels.
    """
    # --- 1. Data Preparation ---
    print("Starting RFM analysis...")
    local_df = df.copy()
    required_cols = [customer_id_col, invoice_date_col, revenue_col]
    assert all(col in local_df.columns for col in required_cols), \
        f"Error: DataFrame must contain the columns: {required_cols}"
    local_df[invoice_date_col] = pd.to_datetime(local_df[invoice_date_col], format=date_format)
    print("Converted date column to datetime objects.")

    # --- 2. Calculate RFM Metrics ---
    analysis_date = local_df[invoice_date_col].max() + pd.Timedelta(days=1)
    recency = local_df.groupby(customer_id_col)[invoice_date_col].max().apply(lambda x: (analysis_date - x).days)
    frequency = local_df.groupby(customer_id_col)[invoice_date_col].count()
    monetary = local_df.groupby(customer_id_col)[revenue_col].sum()
    rfm = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
    print("Calculated Recency, Frequency, and Monetary metrics.")

    # --- 3. K-Means Clustering ---
    if perform_kmeans:
        print("\nPerforming K-Means clustering...")
        rfm_features = rfm[["Recency", "Frequency", "Monetary"]]
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_features)

        # Elbow Method to find the optimal number of clusters (K)
        if generate_plots:
            wcss = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(rfm_scaled)
                wcss.append(km.inertia_)
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, 11), wcss, marker="o")
            plt.title("Elbow Method for Optimal K")
            plt.xlabel("Number of Clusters")
            plt.ylabel("WCSS")
            plt.show()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm["KMeans_Cluster"] = kmeans.fit_predict(rfm_scaled)
        print(f"Assigned customers to {n_clusters} K-Means clusters.")

        # --- 4. Analyze and Interpret K-Means Clusters ---
        print("\n--- K-Means Cluster Analysis & Business Insights ---")
        # Calculate the average RFM values for each cluster
        cluster_analysis = rfm.groupby('KMeans_Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)

        # Sort clusters to assign meaningful labels (e.g., Best Customers, At-Risk)
        # This logic assumes 'Best' customers have low recency, high frequency/monetary
        cluster_analysis = cluster_analysis.sort_values(by=['Recency', 'Monetary'], ascending=[True, False])
        
        # Assign descriptive labels based on sorted order
        cluster_personas = {
            cluster_analysis.index[0]: 'Best Customers (Champions)',
            cluster_analysis.index[1]: 'Potential Loyalists',
            cluster_analysis.index[2]: 'At-Risk Customers',
            cluster_analysis.index[3]: 'Hibernating / Low-Value'
        }
        
        # Add the persona to the analysis table
        cluster_analysis['Persona'] = cluster_analysis.index.map(cluster_personas)
        
        print(cluster_analysis)

        # Map the persona back to the main RFM dataframe for context
        rfm['Cluster_Persona'] = rfm['KMeans_Cluster'].map(cluster_personas)
        
        print("\nBusiness Actions Suggested by Clusters:")
        print("- Best Customers: Nurture with loyalty programs, exclusive offers, and early access.")
        print("- Potential Loyalists: Engage with personalized recommendations and incentives to increase frequency.")
        print("- At-Risk Customers: Launch win-back campaigns with special discounts to re-engage them.")
        print("- Hibernating / Low-Value: Include in general marketing; avoid high-cost campaigns.")


    # --- 5. Save and Visualize ---
    if output_csv_path:
        rfm.to_csv(output_csv_path, index=True)
        print(f"\n✅ Analysis complete. Results saved to: {output_csv_path}")

    if generate_plots and perform_kmeans:
        print("\nGenerating K-Means visualizations...")
        # Scatter plot for K-Means clusters if performed
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="Cluster_Persona", palette="tab10", s=80, alpha=0.8)
        plt.title("K-Means Customer Segments (Recency vs. Monetary)")
        plt.xlabel("Recency (Days since last purchase)")
        plt.ylabel("Monetary (Total spending)")
        plt.legend(title="Customer Persona")
        plt.grid(True)
        plt.show()

    return rfm

# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    # 1. Load your data from a CSV file
    file_path = '/content/drive/MyDrive/refine_file.csv'
    input_df = pd.read_csv(file_path)

    # 2. Call the main function to run the analysis
    rfm_results_df = perform_rfm_analysis(
        df=input_df,
        output_csv_path="customer_segmentation_kmeans_results.csv",
        generate_plots=True,
        perform_kmeans=True,
        n_clusters=4
    )

    # 3. Display the head of the resulting DataFrame
    print("\n--- K-Means Analysis Output (First 5 Rows) ---")
    print(rfm_results_df.head())
