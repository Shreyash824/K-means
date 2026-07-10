import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import CustomerSegmentation

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation using K-Means")

st.markdown("Upload a customer dataset and visualize customer segments.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=3
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

# -----------------------------
# Load Data
# -----------------------------
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Train Model
    # -----------------------------
    segmentation = CustomerSegmentation(
        n_clusters=clusters
    )

    clustered_df, centers, inertia = segmentation.fit(df)

    # -----------------------------
    # Scatter Plot
    # -----------------------------
    st.subheader("Customer Segmentation")

    fig = px.scatter(
        clustered_df,
        x="Annual_Spending",
        y="Order_Count",
        color=clustered_df["Cluster"].astype(str),
        hover_data=["Customer_ID"],
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker=dict(
                size=16,
                color="black",
                symbol="x"
            ),
            name="Centroids"
        )
    )

    fig.update_layout(
        height=600,
        xaxis_title="Annual Spending",
        yaxis_title="Order Count"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Elbow Method
    # -----------------------------
    st.subheader("Elbow Method")

    inertia_values = segmentation.elbow_method(df)

    elbow_df = pd.DataFrame({
        "Clusters": range(1, len(inertia_values)+1),
        "WCSS": inertia_values
    })

    elbow_fig = px.line(
        elbow_df,
        x="Clusters",
        y="WCSS",
        markers=True
    )

    st.plotly_chart(elbow_fig, use_container_width=True)

    # -----------------------------
    # Cluster Distribution
    # -----------------------------
    st.subheader("Cluster Distribution")

    cluster_counts = (
        clustered_df["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = [
        "Cluster",
        "Customers"
    ]

    bar_fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Customers",
        color="Cluster",
        text="Customers"
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    # -----------------------------
    # Download Result
    # -----------------------------
    csv = clustered_df.to_csv(index=False)

    st.download_button(
        "⬇ Download Clustered Dataset",
        csv,
        "clustered_customers.csv",
        "text/csv"
    )

else:

    st.info("Upload a CSV or Excel file to begin.")

