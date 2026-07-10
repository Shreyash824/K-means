import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model import CustomerSegmentation

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")
st.markdown(
    """
    This dashboard performs **Customer Segmentation**
    using the **K-Means Clustering** algorithm.
    """
)

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------
st.sidebar.header("⚙ Model Configuration")


uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

# ----------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Loaded Successfully")

    # ------------------------------------------
    # DATA PREVIEW
    # ------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # ------------------------------------------
    # KPI CARDS
    # ------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
        "Average Spending",
        f"₹{df['Annual_Spending'].mean():,.0f}"
    )

    col3.metric(
        "Average Orders",
        round(df["Order_Count"].mean(), 1)
    )

    st.divider()

    # ------------------------------------------
    # RUN MODEL
    # ------------------------------------------

    model = CustomerSegmentation(
        n_clusters=clusters
    )

    clustered_df, centers, inertia = model.fit(df)

    # ------------------------------------------
    # CUSTOMER SEGMENTATION
    # ------------------------------------------

    st.subheader("Customer Segmentation")

    fig = px.scatter(
        clustered_df,
        x="Annual_Spending",
        y="Order_Count",
        color=clustered_df["Cluster"].astype(str),
        hover_data=["Customer_ID"],
        title="Customer Segmentation",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    # Add Centroids
    fig.add_trace(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker=dict(
                color="black",
                size=18,
                symbol="x"
            ),
            name="Centroids"
        )
    )

    fig.update_layout(
        height=650,
        legend_title="Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
#
    # ----------------------------------------------------
    # ELBOW METHOD
    # ----------------------------------------------------
    st.divider()
    st.subheader("📈 Elbow Method")

    inertia_values = model.elbow_method(df)

    elbow_df = pd.DataFrame({
        "Clusters": list(range(1, len(inertia_values) + 1)),
        "WCSS": inertia_values
    })

    elbow_fig = px.line(
        elbow_df,
        x="Clusters",
        y="WCSS",
        markers=True,
        title="Elbow Method"
    )

    st.plotly_chart(
        elbow_fig,
        use_container_width=True
    )

    # ----------------------------------------------------
    # CLUSTER DISTRIBUTION
    # ----------------------------------------------------
    st.divider()
    st.subheader("📊 Cluster Distribution")

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
        text="Customers",
        title="Customers per Cluster"
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True
    )

    # ----------------------------------------------------
    # CLUSTER STATISTICS
    # ----------------------------------------------------
    st.divider()
    st.subheader("📋 Cluster Statistics")

    summary = (
        clustered_df
        .groupby("Cluster")
        .agg(
            Customers=("Customer_ID", "count"),
            Avg_Spending=("Annual_Spending", "mean"),
            Avg_Orders=("Order_Count", "mean")
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    # ----------------------------------------------------
    # BUSINESS INSIGHTS
    # ----------------------------------------------------
    st.divider()
    st.subheader("💡 Business Insights")

    avg_spending = clustered_df["Annual_Spending"].mean()
    avg_orders = clustered_df["Order_Count"].mean()

    for _, row in summary.iterrows():

        spending = row["Avg_Spending"]
        orders = row["Avg_Orders"]

        if spending >= avg_spending and orders >= avg_orders:

            title = "💎 High-Value Loyal Customers"

            recommendation = """
            • Premium Membership

            • Exclusive Discounts

            • VIP Support

            • Early Product Access
            """

        elif spending >= avg_spending:

            title = "💰 High Spending Customers"

            recommendation = """
            • Loyalty Rewards

            • Personalized Recommendations

            • Upselling Campaigns
            """

        elif orders >= avg_orders:

            title = "🛒 Frequent Buyers"

            recommendation = """
            • Bundle Products

            • Cross Selling

            • Reward Repeat Purchases
            """

        else:

            title = "🌱 Low Engagement Customers"

            recommendation = """
            • Discount Coupons

            • Email Campaigns

            • Seasonal Promotions
            """

        with st.expander(f"Cluster {int(row['Cluster'])} - {title}"):

            st.metric(
                "Customers",
                int(row["Customers"])
            )

            st.metric(
                "Average Spending",
                f"₹{row['Avg_Spending']:,.2f}"
            )

            st.metric(
                "Average Orders",
                f"{row['Avg_Orders']:.2f}"
            )

            st.info(recommendation)

    # ----------------------------------------------------
    # DOWNLOAD DATA
    # ----------------------------------------------------
    st.divider()

    csv = clustered_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Clustered Dataset",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )
