import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, percentileofscore
import numpy as np

def pdf_classification(df, peak, chosen_k, name, output_path):

    fig, ax = plt.subplots(chosen_k//2 + 1, 2, figsize=(10, 8))

    max_pdf_seen = 0
    best_cluster = None

    # print(df.columns)

    for k in range(chosen_k):
        cluster = df.loc[df['cluster'] == k]
        cluster = cluster.drop(['cluster'], axis=1)
        peaks = cluster.max(axis=1)
        ax[k//2, k%2].hist(peaks, bins=50, density=True, label='histogram')

        x = np.linspace(np.min(peaks), np.max(peaks), 1000)
        kde = gaussian_kde(peaks)

        ax[k//2, k%2].plot(x, kde.pdf(x), label='kde')
        ax[k//2, k%2].axvline(x=peak, color='red', label='input peak')
        ax[k//2, k%2].legend(loc='upper right')
        ax[k//2, k%2].set_title(f'Cluster {k}')

        ax[k//2, k%2].set_xlabel("Demand (MW)")
        ax[k//2, k%2].set_ylabel("Density")

        peak_pdf = kde.pdf(peak)

        if peak_pdf > max_pdf_seen:
            max_pdf_seen = peak_pdf
            best_cluster = k

        print(f'{name} Cluster {k} has Density = {peak_pdf} @ {peak} MW')

    # fig.tight_layout()
    # plt.savefig(f'{output_path}/{name}/cluster_pdfs', facecolor='white', transparent=False)
    # plt.close('all')

    # print(f'{name} PDF Classification Graph written to: {output_path}/{name}/cluster_pdfs')

    return best_cluster


def get_sample_load_profiles(df_demand, input_peak, percentile):

    peaks = df_demand.max(axis=1)
    avg_peak = np.mean(peaks)

    # Peaks within +- quantile % 
    input_peak_percentile = percentileofscore(peaks, input_peak)

    lower_bound = input_peak_percentile - percentile
    upper_bound = input_peak_percentile + percentile

    df_demand['percentile'] = percentileofscore(peaks, peaks)

    sample_loads = df_demand.loc[df_demand["percentile"].between(lower_bound, upper_bound)]

    sample_loads = sample_loads.drop('percentile', axis=1)

    # graph_df = sample_loads.reset_index()
    # melted = graph_df.melt(id_vars='date')
    # plt.plot(melted["hour"], melted["value"], 'o', alpha=0.05, color='yellowgreen')

    print(f'{len(sample_loads)} Sample Loads')

    # plt.title(f'Sample Loads for Peak={input_peak} MW')
    # plt.savefig(f'{output_path}/{name}/{output_tag}{"_" if output_tag else ""}sample_loads', facecolor='white', transparent=False)
    # plt.close('all')

    return sample_loads    
