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

    # Two graphs per cluster
    # Can have 2 graphs, 1 cluster per row
    # plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(25, 8))
    gs = GridSpec(chosen_k//2 + 1, 4, figure=fig)

    '''
    in each graph:

    - get sub-df for specified clusters
    - add bar chart subplots at gs[graph idx] with count of day types
    - add scatterplot, plotting all of the load profiles in cluster
    - plot centroid of the same cluster on the same scatterplot


    '''
    for cluster_num, gs_idx in zip(range(chosen_k), range(0, 2*chosen_k, 2)):
        # gs_idx is the index for gridspec
        # cluster_num is the cluster label being investigated

        sub_arr = df.loc[df['cluster'] == cluster_num]

        color = cm.nipy_spectral(np.arange(8).astype(float)/8)

        day_counts = pd.DataFrame(sub_arr['day type'].value_counts())
        day_counts.columns = ['count']

        gs_bar_idx = gs_idx
        gs_scatter_idx = gs_idx + 1

        ax = fig.add_subplot(gs[cluster_num//2, gs_bar_idx % 4])
        ax.barh(y=day_counts.index, width=day_counts['count'], color=color)
        ax.set_title(f'Cluster {cluster_num}')

        ax2 = fig.add_subplot(gs[cluster_num//2, gs_scatter_idx % 4])
        ax2.grid()
        sub_arr = sub_arr.drop(['cluster', 'day type'], axis=1)
        sub_arr.reset_index(inplace=True)
        melted = sub_arr.melt(id_vars='date')
        ax2.scatter(melted["hour"], melted["value"], alpha=0.05, color='yellowgreen')
        ax2.plot(centroids[cluster_num][0:24], 'd-', color='cornflowerblue', linewidth=2.0)
        ax2.set_title(f'Number Profiles: {len(sub_arr)}')
        ax2.set_ylabel('MW')
        ax2.set_xlabel('Hour of Day')

    fig.tight_layout()
    fig.savefig(f'{output_path}/{name}/all_clusters_analysis', facecolor='white', transparent=False)
    plt.close('all')

    print(f'{name} Analysis Graph Generated at: {output_path}/{name}/all_clusters_analysis')