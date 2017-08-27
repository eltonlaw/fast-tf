#!/bin/bash

MODEL="test_experiment"
# Makes a master logs directory if it doesn't already exist
mkdir -p logs
# Makes new log directory to place all experiment outputs
CURRENT_DATE=$(date "+%Y-%m-%d-%H_%M_%S")
LOG_DIR="logs/$CURRENT_DATE"
mkdir $LOG_DIR

# Run Experiments
python3 model_runner.py \
    --log_dir="$LOG_DIR" \
    --model="$MODEL" 


echo "Model checkpoints and logs saved to './logs/$CURRENT_DATE'"
echo "== Job Complete == "
