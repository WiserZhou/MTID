#!/bin/bash
# conda activate MTID

# Configuration
BASE_CMD="python main_distributed.py --dataset=crosstask_how --base_model=predictor --horizon=3"
SEEDS=(3407 3414 3411 3412)
MODEL_DIM=256
NUM_GPUS=8
OUTPUT_DIR="out"

# Ensure output directory exists
# mkdir -p "$OUTPUT_DIR"

# Run experiments
for ((i=0; i<${#SEEDS[@]}; i++)); do
    SEED=${SEEDS[i]}
    GPU=$((i % NUM_GPUS))
    
    EXPERIMENT_NAME="coinpredictor_s${SEED}_d${MODEL_DIM}_h3"
    LOG_FILE="$OUTPUT_DIR/output_${EXPERIMENT_NAME}.log"
    
    nohup $BASE_CMD --name=$EXPERIMENT_NAME --gpu=$GPU \
            --seed=$SEED --model_dim=$MODEL_DIM \
            > "$LOG_FILE" 2>&1 &
    
    echo "Started experiment $((i + 1)) on GPU $GPU with seed $SEED and model_dim $MODEL_DIM"
done

echo "All ${#SEEDS[@]} experiments started. Check individual log files in $OUTPUT_DIR for progress."