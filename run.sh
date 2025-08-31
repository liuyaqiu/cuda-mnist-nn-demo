#!/bin/bash

# CUDA MNIST Neural Network Demo Runner
# This script provides a convenient way to run the model demo with optimized parameters

# Optimized parameters that achieved 81.82% validation accuracy and 80.69% test accuracy
OPTIMIZED_ARGS="--random_seed 42 --epochs 20 --batch_size 128 --learning_rate 0.001 --max_gradient_norm 10.0 --weight_decay 0.0001"

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "This script runs the model demo with optimized parameters."
    echo "You can override by passing your own arguments."
    echo ""
    echo "Default optimized parameters:"
    echo "  --epochs 20"
    echo "  --batch_size 128" 
    echo "  --learning_rate 0.001"
    echo "  --max_gradient_norm 10.0"
    echo "  --weight_decay 0.0001"
    echo "  --random_seed 42 (for reproducibility)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Run with optimized parameters"
    echo "  $0 --epochs 10               # Override epochs only"
    echo "  $0 --help                    # Show model demo help"
    echo ""
    echo "Alternative: Use make directly:"
    echo "  make run-model-demo ARGS=\"--epochs 10 --batch_size 128\""
}

# Check for help flag
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

# If no arguments provided, use optimized parameters
if [ $# -eq 0 ]; then
    ARGS="$OPTIMIZED_ARGS"
else
    # Use provided arguments
    ARGS="$@"
fi

# Run using make with the arguments
exec make run-model-demo ARGS="$ARGS"
