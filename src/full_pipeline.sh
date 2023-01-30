#!/bin/bash
#
# full_pipeline
#
# Clean, Cluster, and Analyze Data

# Stop on errors, print commands
set -eoux pipefail

# Echo the inputs
echo "Input Directory: $1"
echo "Output Directory: $2"
echo "Output Tag: $3"
echo "Current Working Directory: $PWD"

inp=$1
out=$2
tag=$3

# Clean Data
python3 main.py --input_dir $inp --output_dir "CLEAN_TEMPORARY" --clean --output_tag $tag

# Need to move peaks into input directory of clustering
cp $inp/"peak.json" "CLEAN_TEMPORARY"/"peak.json"

# Cluster Data
python3 main.py --input_dir "CLEAN_TEMPORARY" --output_dir "CLUSTER_TEMPORARY" --clustering "S" --output_tag $tag

# Need to move temperature data into input directory of analysis
mkdir -p "CLUSTER_TEMPORARY"/"temp"
cp -r "CLEAN_TEMPORARY"/"temp" "CLUSTER_TEMPORARY"

# Analyze clusters
python3 main.py --input_dir "CLUSTER_TEMPORARY" --output_dir $out --analysis "temp" --output_tag $tag

# Delete temporary directories
rm -r "CLEAN_TEMPORARY"
rm -r "CLUSTER_TEMPORARY"