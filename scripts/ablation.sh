#!/bin/bash

# Array of module_kind values
module_kinds=("i" "e+i" "i+t" "all")

# Array of GPU numbers
gpus=(4 5 6 7)

for i in "${!module_kinds[@]}"; do
    kind="${module_kinds[i]}"
    gpu="${gpus[i]}"
    
    nohup python main_distributed.py \
        --dataset=crosstask_how \
        --name="howpredictor_${kind}" \
        --gpu=$gpu \
        --base_model=predictor \
        --horizon=3 \
        --seed=3407 \
        --ifMask=True \
        --module_kind="$kind" \
        --encoder_kind=conv \
        > "out/output_howpredictor_${kind}.log" 2>&1 &
    
    echo "Started experiment with module_kind=$kind on GPU $gpu"
done

echo "All experiments launched"


# nohup python main_distributed.py \
#     --dataset=crosstask_how \
#     --name=howpredictor_i+t \
#     --gpu=6 \
#     --base_model=predictor \
#     --horizon=3 \
#     --seed=3407 \
#     --ifMask=True \
#     --module_kind=i+t \
#     > out/output_howpredictor_i+t.log 2>&1 &