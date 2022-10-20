"""Command Line Interface for clustering program"""

import sys
import os
import click
import traceback
import pandas as pd

from clean import create_demand_df
from clustering import kmeans_clustering

@click.command()
@click.option('--input_dir', '-i', default='data', type=click.Path(dir_okay=True), help="Input directory")
@click.option('--output_dir', '-o', default='outputs', type=click.Path(dir_okay=True), help="Output directory")
@click.option('--min_k', '-m', default=2, type=int, help='Minimum number of clusters, must be >2')
@click.option('--max_k', '-k', default=25, type=int, help='Maximum number of clusters, must be >2')
@click.option('--algo', '-a', default='KMeans', type=str, help="Algorithm used for clustering: ['Kmeans', 'MST']")
def main(input_dir, output_dir, min_k, max_k, algo):
    try:

        # Handle errors in commmand line input
        input_path = os.path.join(os.getcwd(), input_dir)
        if not os.path.exists(input_path):
            print(f"ERROR: input_dir '{input_dir}' does not exist. " + 
            f"Note that the path is constructed using the current working directory '{os.getcwd()}'")
            sys.exit(1)

        if max_k < 2:
            print(f'ERROR: max_k must be >2 in order for any clusters to form. Input "{max_k}" is not valid.')  
            sys.exit(1)

        if min_k < 2:
            print(f'ERROR: min_k must be >2 in order for any clusters to form. Input "{min_k}" is not valid.')  
            sys.exit(1)

        if algo not in ['KMeans', 'MST']:
            print(f'ERROR: algo must be one of the specified values ["KMeans", "MST"]. Input "{algo}" is not valid.')
            sys.exit(1)

        # Create output path if it does not exist
        output_path = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # loop through each data set in the data directory
        for filename in os.listdir(input_path):
            # create path for each file name
            if not filename.endswith('csv'): continue
            file_path = os.path.join(input_path, filename)
            name = filename.split('.')[0]

            # read in raw data to a dataframe
            raw_data = pd.read_csv(file_path)
        
            # create the df_demand pandas DataFrame with hour as column, row as date, and value as demand
            df_demand = create_demand_df(raw_data)

            # TODO: call a clustering function based on algo
            if algo == 'KMeans': df_clusters = kmeans_clustering(df_demand, min_k, max_k, name, output_path)
            # else: df_clusters = mst_clustering()

            # TODO: analyze the created clusters: day type distribution, cluster homogeniety
            # output analysis graphs to an image and data to a csv?
            # analyze_clusters(df_clusters)
            

            print(algo, input_dir, output_dir)

    except Exception as e:
        print('something is fucked up: \n\n', traceback.format_exc())


if __name__ == "__main__":
    main()
