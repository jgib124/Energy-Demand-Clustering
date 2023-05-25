"""Command Line Interface for clustering program"""

import sys
import os
import json
import click
import traceback
import pandas as pd

from clean import clean_data
from clustering import cluster_subset
from analysis import analyze_df
from classification import pdf_classification, get_sample_load_profiles
from error_checking import error_check

@click.command()
@click.option('--input_dir', '-i', default='data', type=click.Path(dir_okay=True), help="Input directory for segment being run")
@click.option('--output_dir', '-o', default='outputs', type=click.Path(dir_okay=True), help="Output directory for segment being run")
@click.option('--clean', '-l', is_flag=True, help="Clean the given input and print to output directory")
@click.option('--clustering', '-c', default=None, type=str, help="Algorithm used for clustering: ['S']")
@click.option('--analysis', '-a', default=None, type=str, help="Type of analysis done on final clusters: ['temp', 'day']")
@click.option('--min_k', '-m', default=2, type=int, help='Minimum number of clusters, must be >2 and <= max_k')
@click.option('--max_k', '-k', default=15, type=int, help='Maximum number of clusters, must be >2 and >= min_k')
@click.option('--output_tag', '-x', default=None, type=str, help="Tag string added to all outputs for this run")
def main(input_dir, output_dir,
 clean, clustering, analysis, 
  min_k, max_k, output_tag): 

    try:
        # Checks argument inputs and prints errors to output
        error_check(input_dir, clean, clustering,
         analysis, min_k, max_k)

        # Appends input directory (specified in argument) to CWD
        input_path = os.path.join(os.getcwd(), input_dir)

        # Create output path if it does not exist
        output_path = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Check segment being requested and direct to correct function

        # Clean
        if clean:
            # Reads raw data from input path (Demand - CSV, Temp - CSV)
            # Writes cleaned data to output path
            clean_data(input_path, output_path)

        if clustering:
            # Reads cleaned demand data and peaks from input path (Demand - CSV, Peaks - JSON)
            # Writes clusters, centroids, and  Davies-Bouldin Score Graphs to output path 
            # (Clusters - CSV, Centroids - CSV, DB Score (Graph) - PNG)


            # clean all demand and temp files in input directory
            demand_in_path = os.path.join(input_path, "demand")
            peaks_in_path = os.path.join(input_path, "peaks")
            temp_in_path = os.path.join(input_path, "temp")

            os.makedirs(output_path, exist_ok=True)

            for filename in os.listdir(demand_in_path):
                # only concerned with CSVs here
                if not filename.endswith('csv'): continue

                df_demand = None
                if os.path.exists(os.path.join(demand_in_path, filename)):
                    df_demand = pd.read_csv(os.path.join(demand_in_path, filename))
                else:
                    df_demand = pd.read_csv(os.path.join(demand_in_path, filename.split('.')[0]))

                df_temps = None
                if os.path.exists(os.path.join(temp_in_path, filename)):
                    df_temps = pd.read_csv(os.path.join(temp_in_path, filename))
                else:
                    df_temps = pd.read_csv(os.path.join(temp_in_path, filename.split('.')[0]))

                df_peaks = None
                if os.path.exists(os.path.join(peaks_in_path, filename)):
                    df_peaks = pd.read_csv(os.path.join(peaks_in_path, filename))
                else:
                    df_peaks = pd.read_csv(os.path.join(peaks_in_path, filename.split('.')[0]))                

                ba_name = filename.split('.')[0]
                print(ba_name)
                assert(len(ba_name) > 0)

                if clustering == 'S':
                    hourly_profile = cluster_subset(df_demand, df_peaks, df_temps, 
                     min_k, max_k, ba_name, output_path, output_tag)
            

        if analysis:
            clusters_dir_path = os.path.join(input_path, "clusters")
            centroids_dir_path = os.path.join(input_path, "centroids")
            temps_dir_path = os.path.join(input_path, "temp")

            analysis_out_path = os.path.join(output_path, "analysis")

            if not os.path.exists(analysis_out_path):
                os.mkdir(analysis_out_path)

            for filename in os.listdir(clusters_dir_path):
                # only concerned with CSVs here
                if not filename.endswith('csv'): continue

                name = filename.split('.')[0]
                ba = name.split('_')[0]

                clusters_path = os.path.join(clusters_dir_path, filename)
                centroids_path = os.path.join(centroids_dir_path, filename)
                temps_path = os.path.join(temps_dir_path, ba+".csv")

                # read in raw data to a dataframe
                df_clusters = pd.read_csv(clusters_path)
                df_centroids = pd.read_csv(centroids_path)
                df_temps = pd.read_csv(temps_path)

                df_centroids.rename(columns={"Unnamed: 0" : "Cluster"}, inplace=True)

                analyze_df(df_clusters, df_centroids, df_temps,
                 len(df_centroids), name, analysis_out_path, output_tag)



        # # loop through each data set in the data directory
        # temp_path = os.path.join(input_path, "temp")
        # for filename in os.listdir(demand_path):
        #     # create path for each file name
        #     if not filename.endswith('csv'): continue
        #     file_path = os.path.join(input_path, filename)
        #     name = filename.split('.')[0]

        #     if not os.path.exists(f'{output_path}/{name}'):
        #         os.mkdir(f'{output_path}/{name}')

        #     # read in raw data to a dataframe
        #     raw_data = pd.read_csv(file_path)
        
        #     # create the df_demand pandas DataFrame with hour as column, row as date, and value as demand
        #     df_demand = create_demand_df(raw_data)

        #     # Get all load profiles with a peak demand value within our specified tolerance band (%)
        #     df_samples = get_sample_load_profiles(df_demand, peaks[name], percentile, name, output_path, output_tag)

        #     # Cluster the sample load profiles that were returned
        #     # call requested clustering algorithm
        #     df_clusters = None
        #     centroids = None         
        #     chosen_k = None   

        #     # Cluster the sample load profiles that were returned
        #     if clustering == 'KMeans': df_clusters, centroids, chosen_k = kmeans_clustering(df_samples, min_k, max_k, name, output_path, output_tag)

        #     # output analysis graphs to an image and data to a csv?
        #     analyze_df(df_clusters, centroids, chosen_k, name, output_path, output_tag)

            
    except Exception as e:
        print('There was an exception!!!: \n\n', traceback.format_exc())



if __name__ == "__main__":
    main()
