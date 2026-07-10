import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import CustomerSegmentation

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    segmentation = CustomerSegmentation()

    optimal_k, silhouette, inertia = segmentation.find_optimal_k(df)

    st.success(f"Optimal Number of Clusters : {optimal_k}")

    st.success(f"Silhouette Score : {silhouette:.3f}")

    segmentation = CustomerSegmentation(
        n_clusters=optimal_k
    )

    clustered_df, centers, _ = segmentation.fit(df)

    st.subheader("Customer Segmentation")

    fig = px.scatter(
        clustered_df,
        x="Annual_Spending",
        y="Order_Count",
        color=clustered_df["Cluster"].astype(str),
        hover_data=["Customer_ID"]
    )

    fig.add_trace(
        go.Scatter(
            x=centers[:,0],
            y=centers[:,1],
            mode="markers",
            marker=dict(
                color="black",
                size=18,
                symbol="x"
            ),
            name="Centroids"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Elbow Method")

    elbow_df = pd.DataFrame({
        "Clusters": list(range(2, len(inertia)+2)),
        "WCSS": inertia
    })

    fig2 = px.line(
        elbow_df,
        x="Clusters",
        y="WCSS",
        markers=True
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    csv = clustered_df.to_csv(index=False)

    st.download_button(
        "Download Clustered Dataset",
        csv,
        "clustered_dataset.csv",
        "text/csv"
    )
