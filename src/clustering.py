"""Automated clustering algorithm"""

import os
import datetime
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from jenkspy import JenksNaturalBreaks
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from random import randint
import warnings;   warnings.filterwarnings("ignore")

from classification import pdf_classification, get_sample_load_profiles

# Method 1: Clustering the Subset of Load Profiles
# Method 2: Cluster that Maximizes PDF of Peak Value

# returns 
def jenkins_peaks(df, min_k, max_k, name, output_path, output_tag):
    # TODO: choose number of groups using same DB functionality as below
    # similar to sklearn API
    cluster_stats = pd.DataFrame(index=range(min_k, max_k), columns=["Silhouette", "DB", "Penalties"])

    # df.index = df['date']
    df = df.drop('date', axis=1)
    peaks = df['Peak Demand (MW)'].to_list()

    for k in range(min_k, max_k):
        jnb = JenksNaturalBreaks(k)

        jnb.fit(peaks)

        # Calculate the average Silhouette Score for each cluster
        cluster_stats['Silhouette'][k] = silhouette_score(df, jnb.labels_)

        # Davies Boulding score
        this_db = davies_bouldin_score(df, jnb.labels_)
        cluster_stats['DB'][k] = this_db

        # print(f'{name} Jenkins k = {k}, DB = { this_db }')
        
    db_max = cluster_stats['DB'].max()
    db_min = cluster_stats['DB'].min()
    db_diff = db_max - db_min

    PENALTY_CONSTANT = 0.01
    k_range = np.arange(min_k, max_k)
    penalties = cluster_stats['DB'] + (PENALTY_CONSTANT*db_diff*k_range)

    cluster_stats['Penalties'] = penalties

    plt.plot(cluster_stats['Penalties'], label='DB w/ K Penalties Score')
    # add vertical line where DB score is minimized

    plt.legend()
    plt.title(f'{name} Davies-Boulding Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Boulding w/ Penalties Score')

    score_dir = os.path.join(output_path, "scores")
    if not os.path.exists(score_dir):
        os.mkdir(score_dir)

    plt.savefig(f"{score_dir}/{name}_jenkins_db_scores{'_' if output_tag else ''}{output_tag if output_tag else ''}")
    plt.close("all")

    # return the optimal clustered demand
    opt_k = cluster_stats.loc[cluster_stats['Penalties'] == cluster_stats['Penalties'].min()].index[0]

    print(f'{name} Optimal Number of Breaks, Penalized: {opt_k}')

    opt_jnb = JenksNaturalBreaks(int(opt_k))
    opt_jnb.fit(peaks)
    
    df['label'] = opt_jnb.labels_

    return df, opt_k, opt_jnb.breaks_, opt_jnb


# Reads in demand profiles
# Searches from min_k to max_k for optimal number of clusters based on minimum DB Score
# Returns Dataframe with cluster labels and the centroids of each cluster
# NOTE: can be done with any subset of load profiles that are in the same format
#   as those from clean_data
def kmeans_clustering(df, min_k, max_k, name, output_path, output_tag):
    # NOTE: the random state does not affect the clustering strongly in this application
    cluster_stats = pd.DataFrame(index=range(min_k, max_k), columns=["Silhouette", "DB"])

    df.index = df['date']
    df = df.drop('date', axis=1)

    for k in cluster_stats.index:
        kmeans = KMeans(n_clusters=k, random_state=randint(0, 1000)).fit(df)

        # Calculate the average Silhouette Score for each cluster
        cluster_stats['Silhouette'][k] = silhouette_score(df, kmeans.labels_)

        # Davies Boulding score
        cluster_stats['DB'][k] = davies_bouldin_score(df, kmeans.labels_)
        db = cluster_stats['Silhouette'][k]

        # print(f'{name} KMeans k = {k}, DB = { db }')


        
    plt.plot(cluster_stats['DB'], label='Davies-Bouldin Score')
    # add vertical line where DB score is minimized

    plt.legend()
    plt.title(f'{name} Davies-Boulding Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Boulding Score')

    score_dir = os.path.join(output_path, "scores")
    if not os.path.exists(score_dir):
        os.mkdir(score_dir)

    plt.savefig(f"{score_dir}/{name}_db_scores{'_' if output_tag else ''}{output_tag if output_tag else ''}")
    plt.close("all")

    # cluster_stats.to_csv(f'{ba_path}/{output_tag}{"_" if output_tag else ""}cluster_stats')
    # print(f'{ba} Cluster Stats written to: {ba_path}/{output_tag}{"_" if output_tag else ""}cluster_stats')

    # return the optimal clustered demand
    opt_k = cluster_stats.loc[cluster_stats['DB'] == cluster_stats['DB'].min()].index[0]

    print(f'{name} Optimal K: {opt_k}')

    opt_kmeans = KMeans(n_clusters=opt_k, random_state=randint(0, 1000)).fit(df)
    
    df['cluster'] = opt_kmeans.labels_

    centroids = pd.DataFrame(opt_kmeans.cluster_centers_)
    # centroids = centroids.T

    return df, centroids, opt_k


# Cluster Subset Method
# Subset all demand data, cluster that data
# Write the clustered data and the centroids
def cluster_subset(df_demand, df_peaks,
 percentile, min_k, max_k,
  name, output_path, output_tag):

    hourly_profile = []

    # TODO: jenks natural breaks for peaks here
    # Segment peaks into groups that will be clustered
    # Replaces the get_sample_load_profiles functionality, and is only run once at the beginning og this function
    # Store the centroids of all of these natural break groups
    # Query these centroids for the peaks that match up to extend hourly profile
    # Breaks include the min and max values of the entire range
    df, opt_k, breaks, group_classifier = jenkins_peaks(df_peaks, 3, 100, name, output_path, output_tag)

    # kmeans for peaks within a group
    # Save centroids to a queryable object, 0-indexed for groups
    jenkins_centroids = dict()


    for i in range(len(breaks) - 1):
        lower = breaks[i]
        upper = breaks[i + 1]

        # df_subset = df_demand.where
        jenkins_group = df_demand.loc[df_demand.max(axis=1).between(lower, upper)]

        # All days where the peak is within the selected Jenks group
        # Need to account for situation where Jenks group is smaller
        # Than specified maximum number of clusters
        jenks_max_k = max_k
        if (len(jenkins_group) - 1) < max_k:
            jenks_max_k = len(jenkins_group) - 1

        if jenks_max_k <= min_k:
            jenks_max_k = min_k

        df_clusters, centroids, num_clusters = kmeans_clustering(jenkins_group, 
            min_k, jenks_max_k, name, output_path, output_tag)
        
        cluster_dir = os.path.join(output_path, "clusters")
        if not os.path.exists(cluster_dir):
            os.mkdir(cluster_dir)

        centroids_dir = os.path.join(output_path, "centroids")
        if not os.path.exists(centroids_dir):
            os.mkdir(centroids_dir)

        clusters_path = os.path.join(cluster_dir, f"{name}{'_' if output_tag else ''}{output_tag if output_tag else ''}_jenkins_{i}.csv")
        df_clusters.to_csv(clusters_path)

        centroids_path = os.path.join(centroids_dir, f"{name}{'_' if output_tag else ''}{output_tag if output_tag else ''}_jenkins_{i}.csv")
        centroids.to_csv(centroids_path)

        # Save the centroids in memory to query
        jenkins_centroids[i] = centroids

    # print("CENTROIDS DICTIONARY")
    # print(jenkins_centroids)

    for date, row in df_peaks.iterrows():

        # TODO: decide cluster centroid selection criteria
        # Backcasting Validation -- Choose the cluster with the lowest RMSE with actual

        peak_value = row['Peak Demand (MW)']

        peaks_group = group_classifier.predict(peak_value)

        # print(peaks_group, type(peaks_group))
        # print(int(peaks_group))
        # print(list(peaks_group))

        centroids = jenkins_centroids[int(peaks_group)]

        # get same date row from df_demand
        actual = np.array(df_demand.drop(['date'], axis=1).loc[date])

        best_dist = math.inf
        best_k = None
        for k in range(len(centroids)):
            curr_centroid = np.array(centroids.loc[k])
            distance = euclidean(curr_centroid, actual)
            if distance < best_dist:
                best_dist = distance
                best_k = k

            # TODO: can calculate stats here

        hourly_profile.extend(centroids.loc[k])


    df_subset = df_demand.reset_index()
    df_subset = df_subset.drop(['index', 'date'], axis=1)

    df_subset = np.array(list(chain(*np.array(df_subset))))
    hourly_profile = np.array(hourly_profile)

    # print("ACTUAL: ", df_subset)
    # print("PREDICTED: ", hourly_profile)

    fig = plt.figure()
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)


    ax.plot(hourly_profile, '-', label='Predicted', alpha=0.5, linewidth=2)
    ax.plot(df_subset, '--', label='Actual', alpha=0.6)

    # Calculate RMSE and MAE
    rmse = math.sqrt(mean_squared_error(df_subset, hourly_profile))
    mae = mean_absolute_error(df_subset, hourly_profile)
    mape = mean_absolute_percentage_error(df_subset, hourly_profile)

    ax.text(int(0.75*len(hourly_profile)), np.min(hourly_profile), f'RMSE: {round(rmse, 2)} \nMAE: {round(mae, 2)} \nMAPE: {round(mape, 2)}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    hours = np.arange(len(df_subset))

    # plt.fill_between(hours, hourly_profile, df_subset, where=(hourly_profile > df_subset), alpha=0.15, color='green',
    #                 interpolate=True)
    # plt.fill_between(hours, hourly_profile, df_subset, where=(hourly_profile <= df_subset), alpha=0.15, color='red',
    #                 interpolate=True)

    # for x in range(24, len(hourly_profile), 48):
    #     plt.axvspan(x-24, x, alpha=0.25, color='gray')
    ax.set_title(f'{name} Hourly Profile')
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Hour Since Start")
    plt.legend()

    plt.savefig(os.path.join(cluster_dir, f"{name}_hourly_plot.png"), facecolor='white', transparent=False)

    plt.close('all')

    # print(f"{name} HOURLY PROFILE:", hourly_profile)

    return hourly_profile



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


    clusters_path = os.path.join(cluster_dir, f"{name}{'_' if output_tag else ''}{output_tag}.csv")
    df_subset.to_csv(clusters_path)

    # only one centroid written to file
    centroids_path = os.path.join(centroids_dir, f"{name}{'_' if output_tag else ''}{output_tag}.csv")
    centroids.iloc[[best_cluster]].to_csv(centroids_path)
