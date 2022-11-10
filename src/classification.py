'''
1. get the clustered data
2. for each cluster
    - get the peak value from each day
    - create a KDE for those peak values
    - get the PDF for that value in this distribution
3. Return the cluster that results in the highest PDF value for the input peak demand
'''

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def pdf_classification(df, peaks, chosen_k, name, output_path):

    fig, ax = plt.subplots(chosen_k//2 + 1, 2, figsize=(10, 8))
    peak = peaks[name]

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

        peak_pdf = kde.pdf(peak)

        if peak_pdf > max_pdf_seen:
            max_pdf_seen = peak_pdf
            best_cluster = k

        print(f'{name} Cluster {k} has PDF = {peak_pdf}')

    fig.tight_layout()
    plt.savefig(f'{output_path}/{name}/cluster_pdfs', facecolor='white', transparent=False)
    plt.close('all')

    print(f'{name} PDF Classification Graph written to: {output_path}/{name}/cluster_pdfs')

    return best_cluster
