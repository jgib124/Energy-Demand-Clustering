"""Command Line Interface for clustering program"""

import click
import sys
import os

@click.command()
@click.option('--input_dir', '-i', default='data', type=click.Path(dir_okay=True), help="Input directory")
@click.option('--min_k', '-m', default=2, type=int, help='Minimum number of clusters, must be >2')
@click.option('--max_k', '-k', default=25, type=int, help='Maximum number of clusters, must be >2')
@click.option('--algo', '-a', default='KMeans', type=str, help="Algorithm used for clustering: ['Kmeans', 'MST']")
def main(input_dir, min_k, max_k, algo):
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

        
        # TODO: create the df_demand pandas DataFrame with hour as column, row as date, and value as demand
        # df_demand = create_demand_df()

        # TODO: call a clustering function based on algo
        # if algo=='KMeans': df_clusters = kmeans_clustering()
        # else: df_clusters = mst_clustering()

        # TODO: analyze the created clusters: day type distribution, cluster homogeniety
        # output analysis graphs to an image and data to a csv?
        # analyze_clusters(df_clusters)

        print(input_dir, min_k, max_k, algo)

    except:
        return


if __name__ == "__main__":
    main()
