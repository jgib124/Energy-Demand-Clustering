import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def pdf_classification(df, peak, tolerance, chosen_k, name, output_path):

    fig, ax = plt.subplots(chosen_k//2 + 1, 2, figsize=(10, 8))

    max_pdf_seen = 0
    best_cluster = None

    for k in range(chosen_k):
        cluster = df.loc[df['cluster'] == k]
        cluster = cluster.drop(['cluster', 'day type'], axis=1)
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

    fig.tight_layout()
    plt.savefig(f'{output_path}/{name}/cluster_pdfs', facecolor='white', transparent=False)
    plt.close('all')

    print(f'{name} PDF Classification Graph written to: {output_path}/{name}/cluster_pdfs')

    return best_cluster


def get_sample_load_profiles(df, peak, tolerance, cluster_chosen, name, output_path):
    cluster_loads = df.loc[df['cluster'] == cluster_chosen]
    cluster_loads = cluster_loads.drop(['cluster', 'day type'], axis=1)

    peaks = cluster_loads.max(axis=1)
    avg_peak = np.mean(peaks)

    lower_bound = peak - (tolerance * avg_peak)
    upper_bound = peak + (tolerance * avg_peak)

    cluster_loads['peak'] = peaks

    sample_loads = cluster_loads.loc[cluster_loads['peak'].between(lower_bound, upper_bound)]

    sample_loads = sample_loads.drop('peak', axis=1)

    sample_loads.reset_index(inplace=True)
    melted = sample_loads.melt(id_vars='date')
    plt.plot(melted["hour"], melted["value"], 'o', alpha=0.05, color='yellowgreen')

    print(f'{len(sample_loads)} Sample Loads')

    plt.title(f'Sample Loads for Peak={peak} MW')
    plt.savefig(f'{output_path}/{name}/sample_loads', facecolor='white', transparent=False)

    return sample_loads    
