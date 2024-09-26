# Configuration
BASE_CMD="python main_distributed.py --dataset=crosstask_how --base_model=predictor"
GPUS=(0 1 2 3 4 5 6 7)  # 使用8张GPU
OUTPUT_DIR="out"
SEEDS=(3407 3414)  # 使用两个种子：3407和3414
HORIZON=3  # 固定HORIZON为3

# Run experiments
for SEED in "${SEEDS[@]}"; do
    for TRANSFORMER_NUM in {1..8}; do
        GPU=${GPUS[TRANSFORMER_NUM-1]}
        
        EXPERIMENT_NAME="howscorepredictor_s${SEED}_h${HORIZON}_t${TRANSFORMER_NUM}"
        LOG_FILE="$OUTPUT_DIR/output_${EXPERIMENT_NAME}.log"
        
        nohup $BASE_CMD --name=$EXPERIMENT_NAME --gpu=$GPU \
                --seed=$SEED --horizon=$HORIZON --transformer_num=$TRANSFORMER_NUM \
                > "$LOG_FILE" 2>&1 &
        
        echo "Started experiment for GPU $GPU with horizon $HORIZON, seed $SEED, transformer_num $TRANSFORMER_NUM"
    done
done

# // ... 其余注释掉的代码保持不变 ...


# Configuration
# BASE_CMD="python main_distributed.py --dataset=crosstask_how --base_model=predictor"
# SEEDS=(42 137 256 789 1024 2048 3141 4096)  # 新的随机种子
# GPUS=(0 1 2 3 4 5 6 7)  # 使用8张GPU
# OUTPUT_DIR="out"
# # 256
# # Run experiments
# for ((i=0; i<8; i++)); do
#     SEED=${SEEDS[i]}
#     GPU=${GPUS[i]}
    
#     # Set HORIZON based on the index
#     if [ $i -lt 4 ]; then
#         HORIZON=4
#     else
#         HORIZON=6
#     fi
    
#     EXPERIMENT_NAME="howscorepredictor_s${SEED}_h${HORIZON}"
#     LOG_FILE="$OUTPUT_DIR/output_${EXPERIMENT_NAME}.log"
    
#     nohup $BASE_CMD --name=$EXPERIMENT_NAME --gpu=$GPU \
#             --seed=$SEED --horizon=$HORIZON \
#             > "$LOG_FILE" 2>&1 &
    
#     echo "Started experiment $((i + 1)) on GPU $GPU with horizon $HORIZON, seed $SEED"
# done


# Configuration
# BASE_CMD="python main_distributed.py --dataset=crosstask_how --base_model=predictor"
# GPUS=(0 1 2 3 4 5 6 7)  # 使用8张GPU
# OUTPUT_DIR="out"
# SEEDS=(3407 3414)  # 使用两个种子：3407和3414
# HORIZON=3  # 固定HORIZON为3

# # Run experiments
# GPU_INDEX=0
# for SEED in "${SEEDS[@]}"; do
#     for IE_NUM in {0..5}; do
#         GPU=${GPUS[$GPU_INDEX]}
        
#         EXPERIMENT_NAME="howscorepredictor_s${SEED}_h${HORIZON}_ie${IE_NUM}"
#         LOG_FILE="$OUTPUT_DIR/output_${EXPERIMENT_NAME}.log"
        
#         nohup $BASE_CMD --name=$EXPERIMENT_NAME --gpu=$GPU \
#                 --seed=$SEED --horizon=$HORIZON --ie_num=$IE_NUM \
#                 > "$LOG_FILE" 2>&1 &
        
#         echo "Started experiment for GPU $GPU with horizon $HORIZON, seed $SEED, ie_num $IE_NUM"
        
#         # 更新GPU索引，如果达到最后一个GPU，则重新从0开始
#         GPU_INDEX=$((($GPU_INDEX + 1) % 8))
#     done
# done


