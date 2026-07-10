# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import KMeans


class CustomerSegmentation:

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

    def fit(self, df):

        X = df[["Annual_Spending", "Order_Count"]]

        df = df.copy()

        df["Cluster"] = self.model.fit_predict(X)

        return (
            df,
            self.model.cluster_centers_,
            self.model.inertia_
        )

    def elbow_method(self, df, max_clusters=10):

        X = df[["Annual_Spending", "Order_Count"]]

        inertia = []

        for k in range(1, max_clusters + 1):

            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            model.fit(X)

            inertia.append(model.inertia_)

        return inertia
