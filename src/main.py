"""Command Line Interface for clustering program"""

import sys
import os
import json
import click
import traceback
import pandas as pd

from clean import create_demand_df
from clustering import kmeans_clustering
from analysis import analyze_df
from classification import pdf_classification, get_sample_load_profiles
from error_checking import error_check

@click.command()
@click.option('--input_dir', '-i', default='data', type=click.Path(dir_okay=True), help="Input directory for segment being run")
@click.option('--output_dir', '-o', default='outputs', type=click.Path(dir_okay=True), help="Output directory for segment being run")
@click.option('--clean', '-l', is_flag=True, help="Clean the given input and print to output directory")
@click.option('--clustering', '-c', default=None, type=str, help="Algorithm used for clustering: ['KMeans']")
@click.option('--analysis', '-a', default=None, type=str, help="Type of analysis done on final clusters: ['temp', 'day']")
@click.option('--min_k', '-m', default=2, type=int, help='Minimum number of clusters, must be >2 and <= max_k')
@click.option('--max_k', '-k', default=15, type=int, help='Maximum number of clusters, must be >2 and >= min_k')
@click.option('--percentile', '-p', default=2.5, type=float, help="Percentile range of peak values tryurned from sampling function")
@click.option('--output_tag', '-x', default=None, type=str, help="Tag string added to all outputs for this run")
def main(input_dir, output_dir,
 clean, clustering, analysis, 
  min_k, max_k, percentile, output_tag):

    try:
        error_check(input_dir, clean, clustering,
         analysis, min_k, max_k, percentile)

        input_path = os.path.join(os.getcwd(), input_dir)

        # Create output path if it does not exist
        output_path = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Check segment being requested and direct to correct function
        # TODO: create callable functions for each segment of functionality (clean, cluster, analyze)
        # TODO: everything below needs to be reconfigured!!!

        # Get the peak demand predictions
        with open(f'{input_path}/peak.json') as f:
            peaks = json.load(f)

        # loop through each data set in the data directory
        demand_path = os.path.join(input_path, "demand")
        temp_path = os.path.join(input_path, "temp")
        for filename in os.listdir(demand_path):
            # create path for each file name
            if not filename.endswith('csv'): continue
            file_path = os.path.join(input_path, filename)
            name = filename.split('.')[0]

            if not os.path.exists(f'{output_path}/{name}'):
                os.mkdir(f'{output_path}/{name}')

            # read in raw data to a dataframe
            raw_data = pd.read_csv(file_path)
        
            # create the df_demand pandas DataFrame with hour as column, row as date, and value as demand
            df_demand = create_demand_df(raw_data)

            # Get all load profiles with a peak demand value within our specified tolerance band (%)
            df_samples = get_sample_load_profiles(df_demand, peaks[name], percentile, name, output_path, output_tag)

            # Cluster the sample load profiles that were returned
            # call requested clustering algorithm
            df_clusters = None
            centroids = None         
            chosen_k = None   

            # Cluster the sample load profiles that were returned
            if clustering == 'KMeans': df_clusters, centroids, chosen_k = kmeans_clustering(df_samples, min_k, max_k, name, output_path, output_tag)

            # output analysis graphs to an image and data to a csv?
            analyze_df(df_clusters, centroids, chosen_k, name, output_path, output_tag)

            
    except Exception as e:
        print('There was an exception!!!: \n\n', traceback.format_exc())



if __name__ == "__main__":
    main()
