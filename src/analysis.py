from re import sub
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from matplotlib.gridspec import GridSpec


def season_map(dayofyear):

    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    # winter = everything else

    if dayofyear in spring:
        return 'Spring'
    elif dayofyear in summer:
        return 'Summer'
    elif dayofyear in fall:
        return 'Fall'
    else:
        return 'Winter'


def is_weekday(day):
    if day < 5:
        return 'Weekday'
    else:
        return 'Weekend'


def calc_day_type(df_clusters):
    # add a day type column if it does not exist yet
    if 'day type' in df_clusters.columns:
        return

    # calculate season based on the day of year
    df_clusters['day of year'] = df_clusters.index.dayofyear
    df_clusters['season'] = df_clusters['day of year'].map(season_map)
    df_clusters['weekday'] = df_clusters.index.dayofweek
    df_clusters['weekday type'] = df_clusters['weekday'].map(is_weekday)

    # generate string for day type
    df_clusters['day type'] = df_clusters['season'] + ' ' + df_clusters['weekday type']

    df_clusters.drop(['day of year', 'season', 'weekday', 'weekday type'], axis=1, inplace=True)

    return df_clusters

def analyze_df(df_clusters, centroids, chosen_k, name, output_path):
    # similar to 'all_clusters_analysis.png'
    # day type histogram per cluster & scatterplot of demand profile

    df = calc_day_type(df_clusters)
    # print(df)

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(chosen_k//4 + 1, 4, figure=fig)

    '''
    in each graph:

    - get sub-df for specified clusters
    - add bar chart subplots at gs[graph idx] with count of day types
    - add scatterplot, plotting all of the load profiles in cluster
    - plot centroid of the same cluster on the same scatterplot


    '''
    for (gs_idx, cluster_num) in zip(range(0, chosen_k*2, 2), range(chosen_k)):
        # gs_idx is the index for gridspec
        # cluster_num is the cluster label being investigated

        sub_arr = df.loc[df['cluster'] == cluster_num]

        print(sub_arr)
        print(sub_arr.columns)

        color = cm.nipy_spectral(np.arange(8).astype(float)/8)

        day_counts = pd.DataFrame(sub_arr['day type'].value_counts())
        day_counts.columns = ['count']
        day_counts.reset_index(inplace=True)


        ax = fig.add_subplot(gs[gs_idx])
        ax.barh(y=day_counts.index, width=day_counts['count'], color=color)
        ax.set_title(f'Cluster {cluster_num}')

        ax2 = fig.add_subplot(gs[gs_idx + 1])
        sub_arr.drop(['cluster', 'day type'], axis=1, inplace=True)
        sub_arr.reset_index(inplace=True)
        melted = sub_arr.melt(id_vars='date')
        ax2.scatter(melted["hour"], melted["value"], alpha=0.05)
        ax2.plot(centroids[cluster_num][0:24], '.-', color='orange')
        ax2.set_title(f'Number Profiles: {len(sub_arr)}')
        ax2.set_ylabel('MW')
        ax2.set_xlabel('Hour of Day')

    fig.tight_layout()
    fig.savefig(f'{output_path}/{name}/all_clusters_analysis', facecolor='white', transparent=False)
    fig.close()