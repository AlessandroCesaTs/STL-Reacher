#!/bin/bash

SCRIPT_DIR=$(dirname "$0")   #Get the directory where slurm scripts are located

echo "Training"

bash $SCRIPT_DIR/train.sh

echo "Testing"

bash $SCRIPT_DIR/test.sh