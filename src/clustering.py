"""Automated clustering algorithm"""

import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from random import randint


def kmeans_clustering(df, min_k, max_k, ba, output_path):

    # NOTE: the random state does not affect the clustering strongly in this application
    cluster_stats = pd.DataFrame(index=range(min_k, max_k), columns=["Silhouette", "DB"])

    for k in cluster_stats.index:
        kmeans = KMeans(n_clusters=k, random_state=randint(0, 1000)).fit(df)

        # Calculate the average Silhouette Score for each cluster
        cluster_stats['Silhouette'][k] = silhouette_score(df, kmeans.labels_)

        # Davies Boulding score
        cluster_stats['DB'][k] = davies_bouldin_score(df, kmeans.labels_)
        db = cluster_stats['Silhouette'][k]

        print(f'{ba} KMeans k = {k}, DB = { db }')


        
    plt.plot(cluster_stats['DB'], label='Davies-Bouldin Score')
    # add vertical line where DB score is minimized

    plt.legend()
    plt.title(f'{ba} Davies-Boulding Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Boulding Score')

    ba_path = os.path.join(output_path, ba)

    if not os.path.exists(ba_path):
        os.mkdir(ba_path)

    plt.savefig(f'{ba_path}/db_scores')
    plt.close()
    print(f'{ba} Cluster Validity Graph Generated at: {ba_path}/db_scores')

    cluster_stats.to_csv(f'{ba_path}/cluster_stats')
    print(f'{ba} Cluster Stats written to: {ba_path}/cluster_stats')

    # return the optimal clustered demand
    opt_k = cluster_stats.loc[cluster_stats['DB'] == cluster_stats['DB'].min()].index[0]

    print(f'{ba} Optimal K: {opt_k}')

    opt_kmeans = KMeans(n_clusters=opt_k, random_state=randint(0, 1000)).fit(df)
    
    df['cluster'] = opt_kmeans.labels_

    centroids = pd.DataFrame(opt_kmeans.cluster_centers_)
    centroids = centroids.T

    return df, centroids, opt_k



