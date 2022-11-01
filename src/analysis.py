import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
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

    day_counts = pd.DataFrame(df['day type'].value_counts())
    day_counts.columns = ['count']
    day_counts.reset_index(inplace=True)

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(MAX_K//4 + 1, 4, figure=fig)

    # pasted 

    # for (graph_idx, cluster) in zip(range(0, chosen_k*2, 2), range(chosen_k)):
    #         max_cluster = cluster_spread[['day type', 'count', 'day name', 'profile_cluster']].loc[cluster_spread['profile_cluster']  == cluster]

    #         num_day_types = len(max_cluster['day name'].unique())
    #         color = cm.nipy_spectral(np.arange(8).astype(float)/8)

    #         ax = fig.add_subplot(gs[graph_idx])
    #         ax.barh(y=max_cluster['day name'], width=max_cluster['count'], color=color)
    #         ax.set_title(f'Cluster {cluster}')

    #         print(f'Cluster {cluster}')

    #         ax = fig.add_subplot(gs[graph_idx+1])

    #         df_cluster_loads = df_demand.loc[df_demand["profile_cluster"] == cluster].reset_index()
    #         df_cluster_loads.drop(["profile_cluster", "weekday", "season num", "is weekday", "season"], axis=1, inplace=True)
    #         df_cluster_loads = df_cluster_loads.melt(id_vars="date")
    #         ax.scatter(df_cluster_loads["hour"], df_cluster_loads["value"], alpha=0.05)
    #         ax.plot(centroids[cluster][0:24], '.-', color='orange')
    #         ax.set_title(f'Number Profiles: {len(df_demand.loc[df_demand["profile_cluster"] == cluster])}')
    #         ax.set_ylabel('MW')
    #         ax.set_xlabel('Hour of Day')

    #     fig.tight_layout()
    #     fig.savefig(f'silhouette_graphs/{BA}/all_clusters_analysis', facecolor='white', transparent=False)