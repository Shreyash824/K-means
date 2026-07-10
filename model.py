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
        from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class CustomerSegmentation:

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, df):

        X = df[["Annual_Spending", "Order_Count"]]

        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        df = df.copy()

        df["Cluster"] = model.fit_predict(X)

        return (
            df,
            model.cluster_centers_,
            model.inertia_
        )

    def find_optimal_k(self, df, max_clusters=10):

        X = df[["Annual_Spending", "Order_Count"]]

        best_k = 2
        best_score = -1
        inertia = []

        for k in range(2, max_clusters + 1):

            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            labels = model.fit_predict(X)

            inertia.append(model.inertia_)

            score = silhouette_score(X, labels)

            if score > best_score:
                best_score = score
                best_k = k

        return best_k, best_score, inertia

    
