#!/bin/bash
# conda activate MTID
# Base command
BASE_CMD="python main_distributed.py --dataset=coin --base_model=predictor"

# Array of seeds (2 seeds)
SEEDS=(3407 3414)

# Array of model dimensions
MODEL_DIMS=(128 256 512)

# Array of horizons
HORIZONS=(3 4)

# Run experiments
experiment_count=0
for SEED in "${SEEDS[@]}"
do
    for MODEL_DIM in "${MODEL_DIMS[@]}"
    do
        for HORIZON in "${HORIZONS[@]}"
        do
            experiment_count=$((experiment_count + 1))
            GPU=$((experiment_count % 8))  # Use GPUs 0-7 in a round-robin fashion

            nohup $BASE_CMD --name=coinpredictor_s${SEED}_d${MODEL_DIM}_h${HORIZON} --gpu=$GPU \
                 --horizon=$HORIZON --seed=$SEED --model_dim=$MODEL_DIM \
                 > out/output_coinpredictor_s${SEED}_d${MODEL_DIM}_h${HORIZON}.log 2>&1 &

            echo "Started experiment $experiment_count on GPU $GPU with horizon $HORIZON, seed $SEED, and model_dim $MODEL_DIM"
        done
    done
done

echo "All $experiment_count experiments started. Check individual log files for progress."