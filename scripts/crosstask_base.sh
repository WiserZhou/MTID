#!/bin/bash

for i in {1..4}; do
    gpu=$((i-1))
    horizon=$((i+2))
    nohup python main_distributed.py --dataset=crosstask_base --name=test$i --gpu=$gpu \
        --base_model=predictor --horizon=$horizon --ifMask=True > out/test$i.log 2>&1 &
done

echo "All jobs started. Check the log files in the 'out' directory for progress."