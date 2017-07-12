#!/bin/sh

# Makes a master logs directory if it doesn't already exist
mkdir -p logs
# Makes new log directory to place all experiment outputs
current_date=$(date "+%Y-%m-%d-%H_%M_%S")
log_dir="logs/$current_date"
mkdir $log_dir
# Create a file to aggregate all the accuracy results
touch "$log_dir/results.txt"

# Run Experiments
python3 run.py --batch_size=64 --log_dir=$log_dir
python3 run.py --batch_size=128 --log_dir=$log_dir
python3 run.py --batch_size=256 --log_dir=$log_dir
