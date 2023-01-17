"""Automated clustering algorithm"""

import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from random import randint

from classification import pdf_classification, get_sample_load_profiles

# Method 1: Clustering the Subset of Load Profiles
# Method 2: Cluster that Maximizes PDF of Peak Value

# Reads in demand profiles
# Searches from min_k to max_k for optimal number of clusters based on minimum DB Score
# Returns Dataframe with cluster labels and the centroids of each cluster
# NOTE: can be done with any subset of load profiles that are in the same format
#   as those from clean_data
def kmeans_clustering(df, min_k, max_k, name, output_path, output_tag):
    # NOTE: the random state does not affect the clustering strongly in this application
    cluster_stats = pd.DataFrame(index=range(min_k, max_k), columns=["Silhouette", "DB"])

    print(df)

    df.index = df['date']
    df = df.drop('date', axis=1)

    print(df)

    for k in cluster_stats.index:
        kmeans = KMeans(n_clusters=k, random_state=randint(0, 1000)).fit(df)

        # Calculate the average Silhouette Score for each cluster
        cluster_stats['Silhouette'][k] = silhouette_score(df, kmeans.labels_)

        # Davies Boulding score
        cluster_stats['DB'][k] = davies_bouldin_score(df, kmeans.labels_)
        db = cluster_stats['Silhouette'][k]

        print(f'{name} KMeans k = {k}, DB = { db }')


        
    plt.plot(cluster_stats['DB'], label='Davies-Bouldin Score')
    # add vertical line where DB score is minimized

    plt.legend()
    plt.title(f'{name} Davies-Boulding Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Boulding Score')

    ba_path = os.path.join(output_path, name)

    if not os.path.exists(ba_path):
        os.mkdir(ba_path)

    plt.savefig(f'{ba_path}/{output_tag}{"_" if output_tag else ""}db_scores')
    plt.close("all")
    print(f'{name} Cluster Validity Graph Generated at: {ba_path}/{output_tag}{"_" if output_tag else ""}db_scores')

    # cluster_stats.to_csv(f'{ba_path}/{output_tag}{"_" if output_tag else ""}cluster_stats')
    # print(f'{ba} Cluster Stats written to: {ba_path}/{output_tag}{"_" if output_tag else ""}cluster_stats')

    # return the optimal clustered demand
    opt_k = cluster_stats.loc[cluster_stats['DB'] == cluster_stats['DB'].min()].index[0]

    print(f'{name} Optimal K: {opt_k}')

    opt_kmeans = KMeans(n_clusters=opt_k, random_state=randint(0, 1000)).fit(df)
    
    df['cluster'] = opt_kmeans.labels_

    centroids = pd.DataFrame(opt_kmeans.cluster_centers_)
    centroids = centroids.T

    return df, centroids, opt_k


# Cluster Subset Method
# Subset all demand data, cluster that data
# Write the clustered data and the centroids
def cluster_subset(df_demand, input_peak,
 percentile, min_k, max_k,
  name, output_path, output_tag):

    df_subset = get_sample_load_profiles(df_demand, input_peak, percentile)

    df_clusters, centroids, num_clusters = kmeans_clustering(df_subset, 
     min_k, max_k, name, output_path, output_tag)

    cluster_dir = os.path.join(output_path, "clusters")
    if not os.path.exists(cluster_dir):
        os.mkdir(cluster_dir)

    centroids_dir = os.path.join(output_path, "centroids")
    if not os.path.exists(centroids_dir):
        os.mkdir(centroids_dir)


    clusters_path = os.path.join(cluster_dir, f"{name}{'_' if output_tag else ''}{output_tag}")
    df_clusters.to_csv(clusters_path)

    centroids_path = os.path.join(output_path, f"{name}{'_' if output_tag else ''}{output_tag}")
    centroids.to_csv(centroids_path)



# PDF Classification Method
# Cluster all demand data, assign peak to a particular cluster
# Write the cluster load profiles and the centroid
def classify_cluster(df_demand, input_peak,
 min_k, max_k, name, output_path, output_tag):

    df_clusters, centroids, num_clusters = kmeans_clustering(df_demand, 
     min_k, max_k, name, output_path, output_tag)

    best_cluster = pdf_classification(df_clusters, input_peak, num_clusters, name, output_path)

    df_subset = df_clusters.loc[df_clusters['cluster'] == best_cluster]

    cluster_dir = os.path.join(output_path, "clusters")
    if not os.path.exists(cluster_dir):
        os.mkdir(cluster_dir)

    centroids_dir = os.path.join(output_path, "centroids")
    if not os.path.exists(centroids_dir):
        os.mkdir(centroids_dir)


    clusters_path = os.path.join(cluster_dir, f"{name}{'_' if output_tag else ''}{output_tag}")
    df_subset.to_csv(clusters_path)

    # only one centroid written to file
    centroids_path = os.path.join(output_path, f"{name}{'_' if output_tag else ''}{output_tag}")
    centroids.iloc[[best_cluster]].to_csv(centroids_path)
