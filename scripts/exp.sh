#!/bin/bash
conda activate MTID
# Base command
BASE_CMD="python main_distributed.py --dataset=crosstask_how --base_model=predictor --ifMask=True"

# Array of seeds (extended to 16)
SEEDS=(72814 59106 31655 60481 90720 52944 79705 89499 22380 28529 96746 34857 2715 70461 64354 62645)
# (42 1337 2468 9876 5432 7890 1234 6789 2345 8765 4321 9999 1111 7777 3333 8888)
# 3407 3402 3405 3408 3409 3410 3411 3412 3413 3414 3415 3416 3417 3418 3419 3420
# Run experiments
for i in {1..16}
do
    GPU=$((i % 8))  # Use GPUs 0-7 in a round-robin fashion
    SEED=${SEEDS[$((i-1))]}
    HORIZON=6  # Set horizon to 6 for all experiments

    nohup $BASE_CMD --name=howpredictor$i --gpu=$GPU \
         --horizon=$HORIZON --seed=$SEED > out/output_howpredictor$i.log 2>&1 &

    echo "Started experiment $i on GPU $GPU with horizon $HORIZON and seed $SEED"
done

echo "All 16 experiments started. Check individual log files for progress."