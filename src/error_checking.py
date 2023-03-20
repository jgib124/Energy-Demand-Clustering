import sys
import os
import traceback

def error_check(input_dir, clean, clustering,
 analysis, min_k, max_k, percentile):

    # Handle errors in commmand line input
    # Check that the input directory exists
    input_path = os.path.join(os.getcwd(), input_dir)
    if not os.path.exists(input_path):
        print(f"ERROR: input_dir '{input_dir}' does not exist. " + 
        f"Note that the path is constructed using the current working directory '{os.getcwd()}'")
        sys.exit(1)

    # Check that exactly one segment is requested at a time
    num_segments = sum(bool(e) for e in [clean, clustering, analysis])
    if num_segments != 1:
        print(f"ERROR: {num_segments} segments chosen through command to be run. Out of " +
        "[clean, clustering, analysis], only one can be run at a time.")
        sys.exit(10)


    input_folders = os.listdir(input_dir)
    print(input_folders)

    # checks for cleaning:
    if clean:
        # Check that demand data is in input directory
        if "demand" not in input_folders:
            print("ERROR: the 'demand' folder, containing the demand values for each balancing " + 
            f"authority was not present in {input_dir}")
            sys.exit(1) 


    # error checks for clustering
    if clustering:
        # Check that demand data is in input directory
        if "demand" not in input_folders:
            print("ERROR: the 'demand' folder, containing the demand values for each balancing " + 
            f"authority was not present in {input_dir}")
            sys.exit(1) 

        # Check that the peak data is in input directory
        if "daily_peaks" not in input_folders:
            print("ERROR: peak demand values in the form peak.json were not found in the input " + 
            f"directory {input_dir}")
            sys.exit(1) 

        # Check that the percentile for subsetting is valid
        if (percentile < 0) or (percentile > 100):
            print(f"ERROR: percentile value {percentile} for profile subsetting is not valid." + 
            " Choose from range [0, 100]")

        # Check that the maximum number of clusters is valid
        if max_k < 2:
            print(f'ERROR: max_k must be >2 in order for any clusters to form. Input "{max_k}" is not valid.')  
            sys.exit(1)

        # Check that the minimum number of clusters is valid
        if min_k < 2:
            print(f'ERROR: min_k must be >2 in order for any clusters to form. Input "{min_k}" is not valid.')  
            sys.exit(1)

        # Check that the min and max together are valid:
        if min_k > max_k:
            print("ERROR: min_k value must be less that or equal to max_k value.")
            sys.exit(1)

        # Check that a valid clustering algorithm has been chosen
        if clustering not in ['S', 'P']:
            print(f'ERROR: algo must be one of the specified values ["S", "P"]. Input "{clustering}" is not valid.')
            sys.exit(1)

    # error checks for analysis
    if analysis:
        # If analyzing clusters with temperature, check that temperature data is in input directory
        if (analysis == 'temp') and ("temp" not in input_folders):
            print("ERROR: 'temp' folder, containing the hourly temperatures for each balancing " +
            f"authority, was not present in {input_dir}, and user selected 'temp' as the cluster analysis. " +
            "Please include this data in the input directory or choose 'day' analysis.")
            sys.exit(1)

        if "clusters" not in input_folders:
            print(f"ERROR: cluster data was not found in the input directory {input_dir}")
            sys.exit(1)

        if "centroids" not in input_folders:
            print(f"ERROR: centroid data not found in input directory {input_dir}")
            sys.exit(1)