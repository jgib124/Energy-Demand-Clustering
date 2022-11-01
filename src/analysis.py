import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

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

def analyze_df(df_clusters, name, output_path):
    # similar to 'all_clusters_analysis.png'
    # day type histogram per cluster & scatterplot of demand profile

    df = calc_day_type(df_clusters)

    print(f'{name} day types: ', df['day type'].value_counts())
