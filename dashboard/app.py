import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Import functions from your custom Python scripts ---
# Ensure these files are in the same directory as this app.py file
try:
    from preprocessing_and_eda import preprocess_pipeline, analyze_top_performers_by_revenue, analyze_temporal_trends, plot_geographic_heatmap
    from customer_segmention_using_k_means_clusturing import rfm_kmeans_pipeline
    from dynamic_pricing_model import SegmentForecaster, generate_overall_business_insights
except ImportError as e:
    st.error(f"Error importing necessary scripts: {e}. Please make sure 'preprocessing_and_eda.py', 'customer_segmention_using_k_means_clusturing.py', and 'dynamic_pricing_model.py' are in the same folder as app.py.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Pricing & Segmentation Tool",
    page_icon="📈",
    layout="wide"
)

# --- Helper Function to manage file saving ---
def save_uploaded_file(uploaded_file, path):
    """Saves uploaded file to a specified path."""
    try:
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# --- Sidebar for Inputs and Guidance ---
with st.sidebar:
    st.title("🛠️ Controls & Information")
    st.markdown("Follow the steps below to run the analysis.")

    # Step 1: Main Data File Upload
    st.header("1. Upload Data")
    data_file = st.file_uploader("Upload Sales Data (CSV)", type=["csv"])
    competitor_file = st.file_uploader("Upload Competitor Prices (CSV)", type=["csv"])

    # Step 2: Product Selection
    st.header("2. Select Product")
    if data_file:
        # Load a small sample to get stock codes without loading the whole file
        sample_df = pd.read_csv(data_file, nrows=1000)
        stock_codes = sample_df['StockCode'].unique().tolist()
        # Add a default/recommended stock code to the list
        recommended_codes = ['85099B', '22423', '85123A'] + stock_codes
        stock_code_input = st.selectbox(
            "Select or Enter a Product Stock Code",
            options=list(dict.fromkeys(recommended_codes)) # Remove duplicates
        )
    else:
        stock_code_input = st.text_input("Product Stock Code", value="85099B")

    # Step 3: Guidance Section
    st.header("3. How to Use This App")
    st.info(
        """
        - **Data Processing**: Cleans the data, handles missing values, and removes outliers.
        - **EDA**: Generates key charts and insights about top products, sales trends, and customer locations.
        - **Customer Segment**: Groups customers into meaningful personas (e.g., Champions, At-Risk) based on their buying habits.
        - **Prediction**: Forecasts future sales for the selected product, broken down by each customer segment.
        - **Price Recommendation**: Suggests optimal dynamic prices based on the demand forecast.
        - **Business Insights**: Provides a final summary of actionable strategies.
        """
    )

# --- Main Application Body ---
st.title("📈 Dynamic Pricing and Customer Segmentation Dashboard")

# Initialize session state variables to track progress
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'segmented_data' not in st.session_state:
    st.session_state['segmented_data'] = None
if 'forecast_results' not in st.session_state:
    st.session_state['forecast_results'] = None

# Main workflow starts here, checking for the data file
if data_file:
    st.success(f"Successfully loaded '{data_file.name}'. Please proceed with the steps below.")
    main_df = pd.read_csv(data_file)

    # --- 1. Data Processing Section ---
    st.header("Step 1: Data Processing")
    if st.button("Clean and Preprocess Data"):
        with st.spinner("Processing data... This may take a moment."):
            try:
                # The preprocess_pipeline from your script might expect an Excel file,
                # so we adjust to use the loaded DataFrame.
                # We will simulate the pipeline steps as Streamlit can't show live plots from scripts.
                processed_df = preprocess_pipeline(main_df)
                st.session_state['processed_data'] = processed_df
                st.success("Data processing complete!")
                st.write("Preview of Processed Data:")
                st.dataframe(processed_df.head())
            except Exception as e:
                st.error(f"An error occurred during data processing: {e}")


    # --- 2. EDA Section ---
    if st.session_state['processed_data'] is not None:
        st.header("Step 2: Exploratory Data Analysis (EDA)")
        if st.button("Generate EDA Insights"):
            with st.spinner("Generating visualizations..."):
                try:
                    df = st.session_state['processed_data']
                    st.subheader("Top Performers by Revenue")
                    # Capture plots and display them in Streamlit
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    top_products = df.groupby('Description')['Revenue'].sum().nlargest(10)
                    top_products.plot(kind='barh', ax=ax1, color='skyblue')
                    ax1.set_title('Top 10 Products by Revenue')
                    st.pyplot(fig1)

                    st.subheader("Temporal Trends")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    df.groupby(df['InvoiceDate'].dt.month)['Revenue'].sum().plot(kind='line', ax=ax2, marker='o')
                    ax2.set_title('Monthly Revenue Trend')
                    ax2.set_xlabel('Month')
                    st.pyplot(fig2)

                    st.success("EDA generation complete!")
                except Exception as e:
                    st.error(f"An error occurred during EDA: {e}")


    # --- 3. Customer Segmentation Section ---
    if st.session_state['processed_data'] is not None:
        st.header("Step 3: Customer Segmentation")
        if st.button("Segment Customers using K-Means"):
            with st.spinner("Running RFM analysis and clustering..."):
                try:
                    df = st.session_state['processed_data']
                    # Define the output path for the segmentation results
                    segment_output_path = "customer_segmentation_results.csv"
                    segmented_df = rfm_kmeans_pipeline(
                        df=df,
                        output_csv_path=segment_output_path,
                        n_clusters=4
                    )
                    st.session_state['segmented_data'] = segmented_df
                    st.success(f"Customer segmentation complete! Results saved to '{segment_output_path}'")
                    st.write("Customer Segment Profiles:")
                    segment_summary = segmented_df.groupby('Cluster_Persona').agg({
                        'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'KMeans_Cluster': 'count'
                    }).rename(columns={'KMeans_Cluster': 'Customer_Count'}).round(2)
                    st.dataframe(segment_summary)

                    # Provide download link for the segmentation file
                    with open(segment_output_path, "rb") as file:
                        st.download_button(
                            label="Download Segmentation Results",
                            data=file,
                            file_name=segment_output_path,
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"An error occurred during customer segmentation: {e}")


    # --- 4. Prediction Section ---
    if st.session_state['segmented_data'] is not None:
        st.header("Step 4: Demand Forecasting and Prediction")
        if competitor_file is None:
            st.warning("Please upload the competitor price file to enable predictions.")
        else:
            if st.button("Run Forecasting Models"):
                with st.spinner("Training models and forecasting for each segment... This is intensive and may take several minutes."):
                    try:
                        # Save uploaded files to disk so the script can access them
                        save_uploaded_file(data_file, "sales_data.csv")
                        save_uploaded_file(competitor_file, "competitor_prices.csv")

                        # Merge processed data with segment data
                        processed_df = st.session_state['processed_data']
                        segments_df = st.session_state['segmented_data']
                        merged_df = pd.merge(processed_df, segments_df, on='Customer ID', how='left')
                        merged_df['Cluster_Persona'].fillna('Unknown', inplace=True)

                        all_segment_results = []
                        unique_segments = merged_df['Cluster_Persona'].unique()

                        # Configuration for the dynamic pricing model
                        config = {
                            'date_col': 'InvoiceDate', 'date_format': '%Y-%m-%d %H:%M:%S',
                            'quantity_col': 'Quantity', 'customer_id_col': 'Customer ID',
                            'seq_length': 30, 'forecast_horizon': 1, 'train_split': 0.7,
                            'val_split': 0.15, 'learning_rate': 0.001, 'epochs': 50,
                            'patience': 10, 'future_forecast_days': 15, 'seed': 42,
                            'elasticity': -1.5, 'intensity': 0.1, 'holiday_country': 'UK', 'unit_cost': 1.0
                        }

                        progress_bar = st.progress(0)
                        for i, segment in enumerate(unique_segments):
                            st.write(f"--- Processing Segment: {segment} ---")
                            forecaster = SegmentForecaster(
                                df=merged_df,
                                segment=segment,
                                product_code=stock_code_input,
                                competitor_path="competitor_prices.csv",
                                config=config
                            )
                            result = forecaster.run()
                            if result:
                                all_segment_results.append(result)
                                # Capture and display the plot for each segment
                                fig, ax = plt.gcf(), plt.gca()
                                st.pyplot(fig)
                            progress_bar.progress((i + 1) / len(unique_segments))

                        st.session_state['forecast_results'] = all_segment_results
                        st.success("Forecasting complete for all segments!")

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")


    # --- 5. Price Recommendation Section ---
    if st.session_state['forecast_results']:
        st.header("Step 5: Dynamic Price Recommendations")
        results = st.session_state['forecast_results']
        
        # Create a select box to view recommendations by segment
        segment_names = [res['segment'] for res in results]
        selected_segment = st.selectbox("View Price Recommendations for Segment:", segment_names)
        
        for res in results:
            if res['segment'] == selected_segment:
                st.subheader(f"Recommendations for '{res['segment']}'")
                st.dataframe(res['forecast_df'][['Future_Forecast', 'Price_Recommendation']].round(2))
                
                fig, ax = plt.subplots(figsize=(12, 6))
                res['forecast_df']['Price_Recommendation'].plot(ax=ax, marker='o', color='red')
                ax.set_title(f'Price Recommendation Trend for {res["segment"]}')
                ax.set_ylabel("Recommended Price ($)")
                ax.grid(True)
                st.pyplot(fig)


    # --- 6. Business Insights Section ---
    if st.session_state['forecast_results']:
        st.header("Step 6: Final Business Insights")
        if st.button("Generate Overall Business Insights"):
            with st.spinner("Compiling final insights..."):
                try:
                    # The function in your script prints to console. We will capture this logic
                    # and display it nicely in Streamlit.
                    results_list = st.session_state['forecast_results']
                    summary_df = pd.DataFrame(results_list)
                    
                    if not summary_df.empty:
                        st.subheader("Cross-Segment Performance Summary")
                        display_cols = ['segment', 'mae', 'total_forecast_demand', 'avg_recommended_price']
                        st.dataframe(summary_df[display_cols].round(2))

                        highest_demand_segment = summary_df.loc[summary_df['total_forecast_demand'].idxmax()]
                        highest_price_segment = summary_df.loc[summary_df['avg_recommended_price'].idxmax()]

                        st.subheader("Strategic Recommendations")
                        st.info(
                            f"""
                            **📈 Highest Demand Segment**: **'{highest_demand_segment['segment']}'** is predicted to be your largest market for product `{stock_code_input}`.
                            - **Action**: Ensure sufficient inventory and focus marketing efforts on this group.

                            **💰 Highest Price Tolerance**: **'{highest_price_segment['segment']}'** can support the highest prices.
                            - **Action**: Consider offering premium versions, bundles, or value-added services to this segment to maximize revenue.
                            """
                        )
                except Exception as e:
                    st.error(f"An error occurred generating insights: {e}")

else:
    st.info("Please upload a sales data file to begin the analysis.")

