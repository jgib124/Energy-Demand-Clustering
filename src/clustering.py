"""Automated clustering algorithm"""

# Basic Libraries
import os
import datetime
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Specific Tools
from itertools import chain
from numpy.random import normal
from jenkspy import JenksNaturalBreaks
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from random import randint, choice

# Suppress "Future Warnings"
import warnings;   warnings.filterwarnings("ignore")

# Classification Functionality
from classification import pdf_classification, get_sample_load_profiles


# Jenks Natural Breaks algorithm to pre-bin peak values 
# Github: https://github.com/mthh/jenkspy
# 
# k optimized by DB score including a weighted penalty for the number of groups 
#   meant to choose lowest number of groups that pre-bins peaks well
# 
# Input: dataframe of cleaned demand values, k-range
# Writes: Penalized DB Score Charts to scores directory
# Outputs: Jenks groups, Boundary values, Classifier (trained model)
def jenkins_peaks(df, min_k, max_k, name, output_path, output_tag, PENALTY_CONSTANT=0.01):
    # similar to sklearn API
    # Track scores for range of k values (number of groups)
    # NOTE: the "Penalized" score includes a weighted penalty for increasing number of groups
    cluster_stats = pd.DataFrame(index=range(min_k, max_k), columns=["Silhouette", "DB", "Penalized"])

    # Drop dates and make a list of the peak values to use for model
    df = df.drop('date', axis=1)
    peaks = df['Peak Demand (MW)'].to_list()

    # Generate statistics for each K
    for k in range(min_k, max_k):
        # Instantiate and fit Jenks model
        jnb = JenksNaturalBreaks(k)
        jnb.fit(peaks)

        # Calculate the Silhouette Score for each cluster
        cluster_stats['Silhouette'][k] = silhouette_score(df, jnb.labels_)

        # Davies Boulding score
        this_db = davies_bouldin_score(df, jnb.labels_)
        cluster_stats['DB'][k] = this_db
        
    # NOTE: the penalty for the number of groups is specific to each BA
    # Each BA will have a unique range of DB scores
    # In order to make the weighting have an equal affect across BAs, 
    #   penalties are assessed as a proportion of the mean DB score
    db_mean = cluster_stats['DB'].mean()

    # Manually generate scores with assessed penalty
    k_range = np.arange(min_k, max_k)
    penalties = cluster_stats['DB'] + (PENALTY_CONSTANT*db_mean*k_range)

    cluster_stats['Penalized'] = penalties

    plt.plot(cluster_stats['Penalized'], label='DB w/ K-Penalized Score')

    # Add vertical line where Penalized Score is minimized
    best_k_penalized = cluster_stats['Penalized'].astype('float').idxmin()
    # best_k_penalized = cluster_stats.loc[cluster_stats['Penalized'] == cluster_stats['Penalized'].min()].index[0]
    plt.axvline(best_k_penalized, label='Best Penalized Score', linestyle="--", color='green')

    plt.legend()
    plt.title(f'{name} Jenks Penalized Davies-Boulding Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Boulding w/ Penalties Score')

    # Make directory to save DB scores
    score_dir = os.path.join(output_path, "scores")
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(f"{score_dir}/PNG/", exist_ok=True)

    plt.savefig(f"{score_dir}/PNG/{name}_jenks_db_scores{'_' if output_tag else ''}{output_tag if output_tag else ''}")
    plt.close("all")

    print(f'{name} Optimal Number of Breaks, Penalized: {best_k_penalized}')

    # Re-generate model with the optimal number of groups
    opt_jnb = JenksNaturalBreaks(int(best_k_penalized))
    opt_jnb.fit(peaks)
    
    # Add group labels to returned dataframe
    df['label'] = opt_jnb.labels_

    # DF with labels, Optimal Num K, Boundary Breaks of Groups, Trained Model
    return df, best_k_penalized, opt_jnb.breaks_, opt_jnb



# Reads in demand profiles
# Searches from min_k to max_k for optimal number of clusters based on minimum DB Score
# Returns Dataframe with cluster labels and the centroids of each cluster
# NOTE: can be done with any subset of load profiles that are in the same format
#   as those from clean_data

# Inputs: Dataframe of cleaned demand values, range of k values
# Writes:
# Outputs: Labeled dataframe, cluster centroids, optimal K
def kmeans_clustering(df, min_k, max_k, name, output_path, output_tag):
    # NOTE: the random state does not affect the clustering strongly in this application
    # Track scoring statistics

    if len(df) < (max_k + 1):
        max_k = len(df) - 1

    cluster_stats = pd.DataFrame(index=np.arange(min_k, max_k), columns=["DB"])
    cluster_stats.index = pd.to_numeric(cluster_stats.index)

    df.index = df['date']
    df = df.drop('date', axis=1)

    # Track scoring statistics for each k in range
    for k in cluster_stats.index:
        kmeans = KMeans(n_clusters=k, random_state=randint(0, 1000)).fit(df)

        # Calculate the Silhouette Score for each cluster
        # cluster_stats['Silhouette'][k] = silhouette_score(df, kmeans.labels_)

        # Davies Boulding score
        cluster_stats['DB'][k] = davies_bouldin_score(df, kmeans.labels_)
        

    plt.plot(cluster_stats['DB'], label='Davies-Bouldin Score')

    # add vertical line where DB score is minimized
    opt_k = cluster_stats['DB'].astype('float').idxmin()
    plt.axvline(opt_k, label='Optimal K', linestyle="--", color='green')

    plt.title(f'{name} KMeans Davies-Boulding Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Boulding Score')
    plt.legend()

    # Make score directory if necessary
    score_dir = os.path.join(output_path, "scores")
    os.makedirs(score_dir, exist_ok=True)
    os.makedirs(f"{score_dir}/PNG/", exist_ok=True)
    os.makedirs(f"{score_dir}/CSV/", exist_ok=True)

    plt.savefig(f"{score_dir}/PNG/{name}_kmeans_db_scores")
    plt.close("all")

    cluster_stats_path = f"{score_dir}/CSV/{name}_cluster_stats{'_' if output_tag else ''}{output_tag if output_tag else ''}.csv"
    cluster_stats.to_csv(cluster_stats_path)
    print(f'{name} Cluster Stats written to: {cluster_stats_path}')

    print(f'{name} Optimal K: {opt_k}')

    # Re-generate optimal model
    opt_kmeans = KMeans(n_clusters=opt_k, random_state=randint(0, 1000)).fit(df)
    
    # Append cluster labels 
    df['cluster'] = opt_kmeans.labels_

    centroids = pd.DataFrame(opt_kmeans.cluster_centers_)

    # Labeled DF, Cluster Centroids, Optimal K (num groups)
    return df, centroids, opt_k


# Cluster Subset Method
# Subset all demand data, cluster that data
# Write the clustered data and the centroids
def cluster_subset(df_demand, df_peaks, df_temps, 
    min_k, max_k, name, output_path, output_tag):

    hourly_profile = []

    # Make directories for this run
    cluster_dir = os.path.join(output_path, "clusters")
    df_dir = os.path.join(cluster_dir, "demand_dataframes")
    centroids_dir = os.path.join(cluster_dir, "centroid_profiles")
    eval_dir = os.path.join(cluster_dir, "evaluation")
    temperature_dir = os.path.join(cluster_dir, "cluster_temperatures")

    os.makedirs(cluster_dir, exist_ok=True)
    os.makedirs(df_dir, exist_ok=True)
    os.makedirs(centroids_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(temperature_dir, exist_ok=True)

    # Ensure dates are datetime objects
    df_demand['date'] = pd.to_datetime(df_demand['date'])
    df_temps['date'] = pd.to_datetime(df_temps['date'])
    df_peaks['date'] = pd.to_datetime(df_peaks['date'])

    # Only use df_demand dates where there is a corresponding temperature record
    df_demand = df_demand.loc[df_demand['date'].isin(df_temps['date'])]
    df_peaks = df_peaks.loc[df_peaks['date'].isin(df_temps['date'])]


    # --- JENKS NATURAL BREAKS ALGORITHM ---
    # Segment peaks into groups that will be clustered
    # Store the centroids and group members of all of these natural break groups
    # Query these centroids for the peaks that match up to extend hourly profile
    # Breaks include the min and max values of the entire range
    df_labeled, opt_k, breaks, group_classifier = jenkins_peaks(df_peaks, 3, 50, name, output_path, output_tag)

    # kmeans for peaks within a group
    # Save centroids to a queryable object, 0-indexed for groups
    # Save cluster members to a queryable object, 0-indexed for groups
    # jenkins_centroids = dict()
    jenkins_samples = dict()

    for i in range(len(breaks) - 1):
        lower = breaks[i]
        upper = breaks[i + 1]

        # df_subset = df_demand.where
        jenks_group = df_demand.loc[df_demand.max(axis=1).between(lower, upper)]

        # All days where the peak is within the selected Jenks group
        # Need to account for situation where Jenks group is smaller
        # Than specified maximum number of clusters
        jenks_max_k = max_k
        if (len(jenks_group) - 1) < max_k:
            jenks_max_k = len(jenks_group) - 1

        if jenks_max_k <= min_k:
            jenks_max_k = min_k


        print(f"\n\nJenks Group {lower} MW - {upper} MW: ")
        jenks_name = f"{name}_{int(lower)}_{int(upper)}_MW"
        print(f"Group Name: {jenks_name}")
        print("Size of Jenks Group: ", len(jenks_group))

        # centroids_dict = {}
        samples_dict = {}
        
        if len(jenks_group) > (min_k + 1):
            df_clusters, centroids, num_clusters = kmeans_clustering(jenks_group, 
                min_k, jenks_max_k, jenks_name, output_path, output_tag)

            # Use df_clusters to match dates and temperatures up with clusters
            # Get average temperature for each cluster

            fig, ax = plt.subplots(math.ceil(num_clusters/2), 2)

            for c in range(num_clusters):
                # get dates for all profiles within this cluster
                cluster_dates = df_clusters.loc[df_clusters['cluster'] == c].index
                c_temps = df_temps.loc[df_temps['date'].isin(cluster_dates)]['Temperature (F)']

                mu = np.mean(c_temps)
                sigma = np.std(c_temps)

                print('\nCluster:', c)
                print("Profiles in Cluster:", len(c_temps))
                print("Mean Temperature:", mu)
                print("Temperature Standard Deviation:", sigma, '\n')

                # NOTE: df_clusters are the clusters WITHIN the Jenks group
                # centroids_dict[round(mu, 4)] = centroids.iloc[c]
                samples_dict[round(mu, 4)] = df_clusters.loc[df_clusters['cluster'] == c]

                # What data type is this?
                print(type(samples_dict[round(mu, 4)]), samples_dict[round(mu, 4)].head(5))

                gauss = normal(mu, sigma, size=250)
                if (math.ceil(num_clusters/2) > 1):
                    ax[c//2, c%2].set_title(f'Cluster {c}')
                    ax[c//2, c%2].set_xlabel("Temperature (F)")
                    sns.histplot(gauss, kde=True, ax=ax[c//2, c%2])

                else:
                    ax[c%2].set_title(f'Cluster {c}')
                    ax[c%2].set_xlabel("Temperature (F)")
                    sns.histplot(gauss, kde=True, ax=ax[c%2])

                    

            plot_name = f"{jenks_name}{'_' if output_tag else ''}{output_tag if output_tag else ''}_cluster_temperatures.png"
            plt.tight_layout()
            plt.suptitle(f'{name} Cluster Temperatures')

            plt.savefig(os.path.join(temperature_dir, plot_name))
            plt.close('all')

        else: 
            this_date = jenks_group.index
            jenks_group = jenks_group.drop('date', axis=1)
            this_temp = df_temps.loc[df_temps['date'] == this_date]['Temperature (F)']
            # centroids_dict[round(this_temp, 4)] = jenks_group
            samples_dict[round(this_temp, 4)] = jenks_group

            print("Only one cluster for this Jenks group",
                   type(samples_dict[round(this_temp, 4)]),
                     samples_dict[round(this_temp, 4)].head(5))


        df_path = os.path.join(df_dir, f"{jenks_name}{'_' if output_tag else ''}{output_tag if output_tag else ''}_kmeans_dataframe.csv")
        df_clusters.to_csv(df_path)

        centroids_path = os.path.join(centroids_dir, f"{jenks_name}{'_' if output_tag else ''}{output_tag if output_tag else ''}_centroids_dataframe.csv")
        centroids.to_csv(centroids_path)

        # Package the dates, the average temperature, and the centroid together

        # Save the centroids in memory to query
        # jenkins_centroids[i] = centroids_dict
        jenkins_samples[i] = samples_dict


    df_peaks['week'] = df_peaks['date'].dt.week
    df_demand['week'] = df_demand['date'].dt.week


    for _, row in df_peaks.iterrows():

        # Backcasting -- Choose choose the profile that is closest to actual
        
        # Forecasting -- 
        # 1. Seasonal Criteria -- 
        #   Choose cluster based on peak AND the season of the date
        # 2. Temperature -- 
        #   Average Peak temperature for that day of the year, choose profile that has the closest temp
        # 3. Continuity --
        #   Choose based on continuity with previous day (min difference with last value of previous day)
        # 4. Random -- 
        #   Select randomly

        peak_value = row['Peak Demand (MW)']
        date = row['date']

        peaks_group = group_classifier.predict(peak_value)

        # centroids_dict = jenkins_centroids[int(peaks_group)]
        cluster_samples = jenkins_samples[int(peaks_group)]

        # get same date row from df_demand
        # actual = np.array(df_demand.drop(['date'], axis=1).loc[date])
        actual_temp = df_temps.loc[df_temps['date'] == date]['Temperature (F)']

        # print(f'ACTUAL: {date} {actual_temp}')

        # Selection Criteria for Backcasting:
        # Choose profile with minimum euclidean distance
        # best_dist = math.inf
        # best_k = None
        # for k in range(len(centroids)):
        #     curr_centroid = np.array(centroids.loc[k])
        #     distance = euclidean(curr_centroid, actual)
        #     if distance < best_dist:
        #         best_dist = distance
        #         best_k = k

        #     # TODO: can calculate stats here

        # selected_profile = centroids.loc[best_k]
        # selected_peak = np.max(selected_profile)
        # scaling_factor = peak_value / selected_peak
        # scaled_profile = scaling_factor * selected_profile

        # Seletion for forecasting: closest temperature
        # best_centroid = None

        # if len(actual_temp) != 0:
        #     best_diff = math.inf
        #     for temp in centroids_dict.keys():
        #         temp_diff = np.abs(float(temp) - actual_temp.iloc[0])
        #         # print(temp_diff, best_diff)
        #         if temp_diff < best_diff:
        #             best_diff = temp_diff
        #             best_centroid = centroids_dict[temp]

        # else: 
        #     # no temperature for that day on file
        #     # randomly choose??
        #     temp = choice(list(centroids_dict.keys()))
        #     best_centroid = centroids_dict[temp]


        # Selection Criteria Including Stochasticity: sample best cluster
        # TODO: sample from the cluster that is the closest temperature
        best_sample = None
        if len(actual_temp) != 0:
            best_diff = math.inf
            for temp in cluster_samples.keys():
                temp_diff = np.abs(float(temp) - actual_temp.iloc[0])
                # print(temp_diff, best_diff)
                if temp_diff < best_diff:
                    best_diff = temp_diff
                    # Randomly choose a sample from the cluster
                    best_sample = cluster_samples[temp].sample(n=1)

        else: 
            # no temperature for that day on file
            # Randomly choose!!!
            temp = choice(list(cluster_samples.keys()))
            best_centroid = cluster_samples[temp].sample(n=1)


        print("BEST SAMPLE:", best_sample)
        best_sample = best_sample.drop(['cluster'], axis=1)
        selected_peak = np.max(best_sample)
        scaling_factor = peak_value / selected_peak
        scaled_profile = scaling_factor * best_sample

        scaled_peak  = np.max(scaled_profile)
        # print("\nPEAKS:", scaled_peak, peak_value)

        hourly_profile.extend(np.array(scaled_profile).flatten().tolist())


    df_subset = df_demand.reset_index()
    dates = df_subset['date'].to_list()

    # TODO: there have been problems with the dates not matching up
    # Add if-statements to check that data structures are the same length
    x_hours = None
    if len(hourly_profile) > len(dates)*24:
        # Hourly profile may include an extra day at the end
        x_hours = pd.date_range(dates[0], pd.to_datetime(dates[-1]) + datetime.timedelta(1), freq='H', inclusive='left')
    else:
        x_hours = pd.date_range(dates[0], pd.to_datetime(dates[-1]), freq='H', inclusive='left')

    print("LENGTHS: ", len(df_subset), len(x_hours), len(hourly_profile), len(dates)*24)
    print("DATES: ", dates[0], dates[-1])
    print("HOURS: ", x_hours[0], x_hours[-1])


    # print(dates)
    # print(x_hours)
    # print(df_subset)

    df_subset = df_subset.drop(['index', 'date', 'week'], axis=1)   

    df_subset = np.array(list(chain(*np.array(df_subset))))
    hourly_profile = np.array(hourly_profile)

    fig = plt.figure()
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)

    ax.plot(x_hours, df_subset, 'o', label='Actual', alpha=0.4)
    ax.plot(x_hours, hourly_profile, 'o', label='Predicted', alpha=0.6)
    plt.gcf().autofmt_xdate()

    for i in np.arange(0, len(x_hours), 48):
        start = x_hours[i]
        end = x_hours[i] + datetime.timedelta(hours=24)

        plt.axvspan(start, end, color='gray', alpha=0.2)

    for i in np.arange(0, len(x_hours), 24):
        start = x_hours[i]
        end = x_hours[i] + datetime.timedelta(hours=24)
        day_peak = np.max(hourly_profile[i:i+24])

        plt.hlines(y=day_peak, xmin=start, xmax=end, linewidth=1, linestyle='--', color='green')


    # Calculate RMSE and MAE
    rmse = math.sqrt(mean_squared_error(df_subset, hourly_profile))
    mae = mean_absolute_error(df_subset, hourly_profile)
    mape = mean_absolute_percentage_error(df_subset, hourly_profile)

    error_values = pd.DataFrame(index=['RMSE', 'MAE', 'MAPE'], columns=['Value'], data=[rmse, mae, mape])
    error_values.to_csv(f"{eval_dir}/{name}_error_values.csv")

    with open(f"{eval_dir}/error_values.csv", 'a+') as f:
        if os.stat(f"{eval_dir}/error_values.csv").st_size == 0:
            f.write("ba,stat,value\n")
        f.write(f"{name},RMSE,{rmse}\n")
        f.write(f"{name},MAE,{mae}\n")
        f.write(f"{name},MAPE,{mape}\n")

    ax.text(x_hours[int(0.75*len(x_hours))], np.min(hourly_profile), f'RMSE: {round(rmse, 2)} \nMAE: {round(mae, 2)} \nMAPE: {round(mape, 2)}', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    hours = np.arange(len(df_subset))

    ax.set_title(f'{name} Hourly Profile')
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Date")

    plt.legend()

    path_8760 = f"{cluster_dir}/8760/"
    os.makedirs(f"{cluster_dir}/8760/", exist_ok=True)

    plt.savefig(f"{path_8760}/{name}_hourly_plot.png", facecolor='white', transparent=False)

    plt.close('all')

    return hourly_profile

