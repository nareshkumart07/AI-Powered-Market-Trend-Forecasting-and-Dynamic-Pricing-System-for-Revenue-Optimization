"""
This script performs a complete RFM (Recency, Frequency, Monetary) analysis
and K-Means clustering on customer data. The code is structured into modular,
reusable functions for each step of the analysis.
"""

# --- 0. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- STEP 1: DATA PREPARATION AND RFM CALCULATION ---

def prepare_rfm_data(
    df: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    revenue_col: str,
    date_format: str
) -> pd.DataFrame:
    """Validates and prepares the input DataFrame for RFM analysis."""
    print("--- Step 1a: Preparing Data ---")
    local_df = df.copy()
    required_cols = [customer_id_col, invoice_date_col, revenue_col]
    assert all(col in local_df.columns for col in required_cols), \
        f"Error: DataFrame must contain the columns: {required_cols}"
    local_df[invoice_date_col] = pd.to_datetime(local_df[invoice_date_col], format=date_format)
    print("-> Date column converted to datetime objects.")
    return local_df

def calculate_rfm_metrics(
    df: pd.DataFrame,
    customer_id_col: str,
    invoice_date_col: str,
    revenue_col: str
) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary values for each customer."""
    print("--- Step 1b: Calculating RFM Metrics ---")
    analysis_date = df[invoice_date_col].max() + pd.Timedelta(days=1)
    recency = df.groupby(customer_id_col)[invoice_date_col].max().apply(lambda x: (analysis_date - x).days)
    frequency = df.groupby(customer_id_col)[invoice_date_col].count()
    monetary = df.groupby(customer_id_col)[revenue_col].sum()
    rfm = pd.DataFrame({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary})
    print("-> RFM metrics calculated.")
    return rfm

# --- STEP 2: K-MEANS MODELING ---

def scale_rfm_data(rfm_df: pd.DataFrame) -> np.ndarray:
    """Scales the RFM features using StandardScaler."""
    print("--- Step 2a: Scaling RFM Data ---")
    rfm_features = rfm_df[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    print("-> Data scaled successfully.")
    return rfm_scaled

def find_optimal_clusters_elbow(rfm_scaled_data: np.ndarray):
    """Plots the Elbow Method curve to help find the optimal number of clusters (K)."""
    print("--- Step 2b: Finding Optimal K (Elbow Method) ---")
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled_data)
        wcss.append(km.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker="o", linestyle="--")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.grid(True)
    plt.show()

def run_kmeans_model(rfm_scaled_data: np.ndarray, n_clusters: int) -> np.ndarray:
    """Fits a K-Means model and returns the cluster labels for each customer."""
    print(f"--- Step 2c: Running K-Means with Optimal Clusters ---")
    print(f"-> Running with {n_clusters} clusters.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rfm_scaled_data)
    print("-> Customers assigned to clusters.")
    return cluster_labels

# --- STEP 3: CLUSTER ANALYSIS AND INTERPRETATION ---

def calculate_cluster_profiles(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the mean RFM values for each cluster."""
    print("--- Step 3a: Calculating Cluster Profiles ---")
    cluster_analysis = rfm_df.groupby('KMeans_Cluster').agg({
        'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
    }).round(2)
    return cluster_analysis

def assign_cluster_personas(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    """Assigns descriptive persona labels to clusters based on their RFM profiles."""
    print("--- Step 3b: Assigning Personas to Clusters ---")
    # Sort clusters to create a consistent ranking (e.g., best to worst)
    sorted_profiles = cluster_profiles.sort_values(by=['Recency', 'Monetary'], ascending=[True, False])
    
    # Heuristic mapping for 4 clusters. This may need adjustment for a different K.
    persona_map = {
        sorted_profiles.index[0]: 'Best Customers (Champions)',
        sorted_profiles.index[1]: 'Potential Loyalists',
        sorted_profiles.index[2]: 'At-Risk Customers',
        sorted_profiles.index[3]: 'Hibernating / Low-Value'
    }
    print("-> Personas assigned based on cluster characteristics.")
    return persona_map

# --- STEP 4: REPORTING AND VISUALIZATION ---

def print_cluster_summary(rfm_df_with_personas: pd.DataFrame):
    """Prints the final cluster analysis table and suggested business actions."""
    print("\n--- K-Means Cluster Analysis & Business Insights ---")
    
    # Create the summary table
    cluster_summary = rfm_df_with_personas.groupby('Cluster_Persona').agg({
        'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'KMeans_Cluster': 'count'
    }).rename(columns={'KMeans_Cluster': 'Customer_Count'}).round(2)
    print(cluster_summary)
    
    print("\n--- Business Actions Suggested by Clusters ---")
    print("- Best Customers: Nurture with loyalty programs, exclusive offers, and early access.")
    print("- Potential Loyalists: Engage with personalized recommendations and incentives to increase frequency.")
    print("- At-Risk Customers: Launch win-back campaigns with special discounts to re-engage them.")
    print("- Hibernating / Low-Value: Include in general marketing; avoid high-cost campaigns.")

def plot_kmeans_clusters(rfm_df: pd.DataFrame):
    """Visualizes the K-Means clusters using a scatter plot."""
    print("\n--- Generating K-Means Visualizations ---")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=rfm_df, x="Recency", y="Monetary", hue="Cluster_Persona", palette="tab10", s=80, alpha=0.8)
    plt.title("K-Means Customer Segments (Recency vs. Monetary)", fontsize=16)
    plt.xlabel("Recency (Days since last purchase)", fontsize=12)
    plt.ylabel("Monetary (Total spending)", fontsize=12)
    plt.legend(title="Customer Persona")
    plt.grid(True)
    plt.show()

# --- 5. MAIN PIPELINE ORCHESTRATOR ---

def rfm_kmeans_pipeline(
    df: pd.DataFrame,
    customer_id_col: str = 'Customer ID',
    invoice_date_col: str = 'InvoiceDate',
    revenue_col: str = 'Revenue',
    date_format: str = '%d-%m-%Y %H:%M',
    output_csv_path: Optional[str] = "customer_segments_kmeans.csv",
    n_clusters: int = 4
) -> pd.DataFrame:
    """Orchestrates the complete RFM and K-Means analysis pipeline."""
    # Step 1: Data Prep & RFM Calculation
    prepared_df = prepare_rfm_data(df, customer_id_col, invoice_date_col, revenue_col, date_format)
    rfm_metrics = calculate_rfm_metrics(prepared_df, customer_id_col, invoice_date_col, revenue_col)
    
    # Step 2: K-Means Modeling
    rfm_scaled = scale_rfm_data(rfm_metrics)
    find_optimal_clusters_elbow(rfm_scaled) # Visual aid for choosing K
    cluster_labels = run_kmeans_model(rfm_scaled, n_clusters)
    rfm_clustered = rfm_metrics.copy()
    rfm_clustered['KMeans_Cluster'] = cluster_labels
    
    # Step 3: Cluster Interpretation
    cluster_profiles = calculate_cluster_profiles(rfm_clustered)
    persona_map = assign_cluster_personas(cluster_profiles)
    rfm_analyzed = rfm_clustered.copy()
    rfm_analyzed['Cluster_Persona'] = rfm_analyzed['KMeans_Cluster'].map(persona_map)
    
    # Step 4: Reporting and Visualization
    print_cluster_summary(rfm_analyzed)
    plot_kmeans_clusters(rfm_analyzed)

    # Step 5: Save Final Output
    if output_csv_path:
        rfm_analyzed.to_csv(output_csv_path, index=True)
        print(f"\n✅ Analysis complete. Results saved to: {output_csv_path}")
        
    return rfm_analyzed

# --- HOW TO USE THE FUNCTION IN A PIPELINE ---
if __name__ == '__main__':
    try:
        file_path = '/content/refine_file.csv'
        input_df = pd.read_csv(file_path)

        rfm_results_df = rfm_kmeans_pipeline(
            df=input_df,
            n_clusters=4
        )

        print("\n--- K-Means Analysis Output (First 5 Rows) ---")
        print(rfm_results_df.head())

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
