#!/bin/bash

# Script to run GNN memorization experiments across multiple datasets and model configurations
# Usage: ./run_experiments.sh

# Create directories if they don't exist
mkdir -p results

# Set common parameters
MODEL="gcn"
EPOCHS=100
HIDDEN_DIM=32
LR=0.01
WEIGHT_DECAY=5e-4
# Datasets to run
DATASETS=("Cora")
#DATASETS=("Chameleon" "Squirrel")

# Number of layers to test
LAYERS=(3)

# Function to run an experiment and log output
run_experiment() {
    dataset=$1
    model=$2
    layers=$3
    
    echo "========================================"
    echo "Running experiment with:"
    echo "Dataset: $dataset"
    echo "Model: $model"
    echo "Layers: $layers"
    echo "========================================"
    
    # Run the experiment
    python main_fixed.py \
        --dataset $dataset \
        --model_type $model \
        --num_layers $layers \
        --hidden_dim $HIDDEN_DIM \
        --epochs $EPOCHS \
        --lr $LR \
        --weight_decay $WEIGHT_DECAY
    
    # Check if experiment was successful
    if [ $? -eq 0 ]; then
        echo "Experiment completed successfully!"
    else
        echo "ERROR: Experiment failed!"
    fi
    echo ""
}

# Run all combinations
total_experiments=$((${#DATASETS[@]} * ${#LAYERS[@]}))
current=1

for dataset in "${DATASETS[@]}"; do
    for layers in "${LAYERS[@]}"; do
        echo "Progress: Experiment $current/$total_experiments"
        run_experiment $dataset $MODEL $layers
        current=$((current + 1))
    done
done

echo "All experiments completed!"