#!/bin/bash
#
# full_pipeline
#
# Clean, Cluster, and Analyze Data

# Stop on errors, print commands
set -eoux pipefail

# Ensure that all necessary libraries are installed
pip install -r requirements.txt

# Echo the inputs
echo "Input Directory: $1"
echo "Output Directory: $2"
echo "Output Tag: $3"
echo "Current Working Directory: $PWD"

inp=$1
out=$2
tag=$3

# Create output directory
mkdir "outputs_$tag" -p

# Clean Data
python3 main.py --input_dir $inp --output_dir "outputs_$tag/Cleaned_$tag" --clean --output_tag $tag

# Need to move peaks into input directory of clustering
# cp $inp/"peak.json" "CLEAN_TEMPORARY"/"peak.json"
cp -r $inp/"peaks" "outputs_$tag/Cleaned_$tag"

# Cluster Data
python3 main.py --input_dir "outputs_$tag/Cleaned_$tag" --output_dir "outputs_$tag/Clustered_$tag" --clustering "S" --output_tag $tag

# Assemble the Year-Long Hourly Profile

# # Need to move temperature data into input directory of analysis
# STEP NOT NEEDED IF NOT DOING ANALYSIS
# mkdir -p "CLUSTER_TEMPORARY"/"temp"
# cp -r "CLEAN_TEMPORARY"/"temp" "CLUSTER_TEMPORARY"

# Analyze clusters
# DON'T NEED TO REPEAT ANALYSIS EVERY TIME
# python3 main.py --input_dir "CLUSTER_TEMPORARY" --output_dir $out --analysis "temp" --output_tag $tag

# Use clusters in CLUSTER_TEMPORARY to generate 24 hour profiles from daily peaks
# Current selection criteria: RANDOM


# Delete temporary directories
# rm -r "CLEAN_TEMPORARY"
# rm -r "CLUSTER_TEMPORARY"