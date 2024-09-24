# Configuration
BASE_CMD="python main_distributed.py --dataset=crosstask_how --base_model=predictor"
HORIZON=5
SEEDS=(3407 3407 3407 3407 3407 3407 5678 5678 5678 5678 5678 5678)
MODEL_DIM=256  # Changed to a single value
GPUS=(0 1 2 3 4 5 6 7 0 1 2 3)  # 循环使用GPU 0-7
OUTPUT_DIR="out"

LOSS_TYPES=("Weighted_MSE" "Weighted_MSE" "Weighted_Gradient_MSE" "Weighted_MSE" "Weighted_MSE" "Weighted_Gradient_MSE")
MASK_LOSSES=(2 2 2 1 1 1)
WEIGHTS=(1 6 6 1 6 6)

# Run experiments
for ((i=0; i<12; i++)); do
    SEED=${SEEDS[i]}
    GPU=${GPUS[i]}
    LOSS_TYPE=${LOSS_TYPES[i % 6]}
    MASK_LOSS=${MASK_LOSSES[i % 6]}
    WEIGHT=${WEIGHTS[i % 6]}
    
    EXPERIMENT_NAME="howscorepredictor_s${SEED}_d${MODEL_DIM}_h${HORIZON}_l${LOSS_TYPE}_m${MASK_LOSS}_w${WEIGHT}"
    LOG_FILE="$OUTPUT_DIR/output_${EXPERIMENT_NAME}.log"
    
    nohup $BASE_CMD --name=$EXPERIMENT_NAME --gpu=$GPU \
            --seed=$SEED --model_dim=$MODEL_DIM --horizon=$HORIZON \
            --loss_type=$LOSS_TYPE --mask_loss=$MASK_LOSS --weight=$WEIGHT \
            > "$LOG_FILE" 2>&1 &
    
    echo "Started experiment $((i + 1)) on GPU $GPU with horizon $HORIZON, seed $SEED, model_dim $MODEL_DIM, loss_type $LOSS_TYPE, mask_loss $MASK_LOSS, and weight $WEIGHT"
done

