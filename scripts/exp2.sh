#!/bin/bash
# conda activate MTID
# Base command
BASE_CMD="python main_distributed.py --dataset=crosstask_how --base_model=predictor --ifMask=True"

# Array of seeds (reduced to 8)
SEEDS=(23498 76301 45872 98654 12345 67890 54321 87654)

# Run experiments
for i in {1..8}
do
    GPU=$((i % 8))  # Use GPUs 0-7 in a round-robin fashion
    SEED=${SEEDS[$((i-1))]}
    HORIZON=6  # Set horizon to 6 for all experiments

    nohup $BASE_CMD --name=howpredictor$i --gpu=$GPU \
         --horizon=$HORIZON --seed=$SEED > out/output_howpredictor$i.log 2>&1 &

    echo "Started experiment $i on GPU $GPU with horizon $HORIZON and seed $SEED"
done

echo "All 8 experiments started. Check individual log files for progress."