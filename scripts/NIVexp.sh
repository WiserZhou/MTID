#!/bin/bash
# conda activate MTID
# Base command
BASE_CMD="python main_distributed.py --dataset=NIV --base_model=predictor --ifMask=False"

# Array of seeds
SEEDS=(72814 59106 31655 60481)

# Array of model dimensions
MODEL_DIMS=(64 128 256)

# Run experiments
experiment_count=0
for model_dim in "${MODEL_DIMS[@]}"
do
    for i in "${!SEEDS[@]}"
    do
        experiment_count=$((experiment_count + 1))
        GPU=$((experiment_count % 8))  # Use GPUs 0-7 in a round-robin fashion
        SEED=${SEEDS[$i]}
        HORIZON=3  # Set horizon to 3 for all experiments

        nohup $BASE_CMD --name=howpredictor${model_dim}_$((i+1)) --gpu=$GPU \
             --horizon=$HORIZON --seed=$SEED --model_dim=$model_dim > out/output_nivpredictor${model_dim}_$((i+1)).log 2>&1 &

        echo "Started experiment ${experiment_count} on GPU $GPU with horizon $HORIZON, seed $SEED, and model_dim $model_dim"
    done
done

echo "All $(( ${#MODEL_DIMS[@]} * ${#SEEDS[@]} )) experiments started. Check individual log files for progress."