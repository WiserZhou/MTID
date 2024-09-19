#!/bin/bash

# Array of GPU IDs to use
gpus=(0 1 2 3 4 5 6 7)
gpu_index=0

# Array of seed values
seeds=(3407 3414)

for horizon in {3..6}; do
  for mask in True False; do
    for seed in "${seeds[@]}"; do
      # Use modulo to cycle through GPUs
      gpu=${gpus[$((gpu_index % ${#gpus[@]}))]}
      
      experiment_name="how_h${horizon}_m${mask}_s${seed}"
      
      nohup python main_distributed.py \
        --dataset=crosstask_how \
        --name=$experiment_name \
        --gpu=$gpu \
        --base_model=predictor \
        --horizon=$horizon \
        --seed=$seed \
        --ifMask=$mask > "out/output_${experiment_name}.log" 2>&1 &
      
      echo "Started experiment: $experiment_name on GPU $gpu"
      
      # Increment GPU index
      ((gpu_index++))
    done
  done
done

echo "All experiments launched"