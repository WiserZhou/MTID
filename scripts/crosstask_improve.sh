# Configuration
BASE_CMD="python main_distributed.py --dataset=NIV --base_model=predictor"
HORIZONS=(3 4 3 4)
SEEDS=(3407 3407 3407 3407)
MODEL_DIMS=(128 128 256 256)
GPUS=(4 5 6 7)  # 更新为指定的GPU编号
OUTPUT_DIR="out"

# ... existing code ...

# Run experiments
for ((i=0; i<${#HORIZONS[@]}; i++)); do
    HORIZON=${HORIZONS[i]}
    SEED=${SEEDS[i]}
    MODEL_DIM=${MODEL_DIMS[i]}
    GPU=${GPUS[i]}
    
    EXPERIMENT_NAME="howscorepredictor_s${SEED}_d${MODEL_DIM}_h${HORIZON}"
    LOG_FILE="$OUTPUT_DIR/output_${EXPERIMENT_NAME}.log"
    
    nohup $BASE_CMD --name=$EXPERIMENT_NAME --gpu=$GPU \
            --seed=$SEED --model_dim=$MODEL_DIM --horizon=$HORIZON \
            > "$LOG_FILE" 2>&1 &
    
    echo "Started experiment $((i + 1)) on GPU $GPU with horizon $HORIZON, seed $SEED, and model_dim $MODEL_DIM"
done

# ... existing code ...