#!/bin/bash

# Makes a master logs directory if it doesn't already exist
mkdir -p logs
# Makes new log directory to place all experiment outputs
current_date=$(date "+%Y-%m-%d-%H_%M_%S")
log_dir="logs/$current_date"
mkdir $log_dir

# Create a file to aggregate all the accuracy results(done with the '>>' below)
# touch "$log_dir/results.txt"

# Run Experiments
python3 model_runner.py --batch_size=64 --log_dir=$log_dir >> $log_dir/results.txt
# python3 model_runner.py --batch_size=128 --log_dir=$log_dir >> $log_dir/results.txt
# python3 model_runner.py --batch_size=256 --log_dir=$log_dir >> $log_dir/results.txt

echo "== Job Complete == "
