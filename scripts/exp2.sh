# Define the parameters
DATASET="crosstask_how"
GPUS=(0 1 2 3 4 5 6 7)  # Define the GPUs to use
BASE_MODEL="predictor"
HORIZON=3
LOG_DIR="out"
SEEDS=(3407 3414)
IFMASK_VALUES=(False True)
MASK_ITERATION_VALUES=("add" "none")
# Loop over the combinations of seeds and ifMask values
GPU_INDEX=0  # Initialize GPU index
for SEED in "${SEEDS[@]}"; do
  for IFMASK in "${IFMASK_VALUES[@]}"; do
    for MASK_ITERATION in "${MASK_ITERATION_VALUES[@]}"; do
        GPU=${GPUS[GPU_INDEX]}  # Assign GPU based on index
        NAME="how1_seed${SEED}_ifMask${IFMASK}_mask_iteration${MASK_ITERATION}"  # Update NAME to be unique
        LOG_FILE="${LOG_DIR}/output_${NAME}.log"  # Update LOG_FILE to be unique
        nohup python main_distributed.py --dataset=${DATASET} --name=${NAME} --gpu=${GPU} \
            --base_model=${BASE_MODEL} --horizon=${HORIZON} --ifMask=${IFMASK} --seed=${SEED} \
            --mask_iteration=${MASK_ITERATION} > ${LOG_FILE} 2>&1 &
        GPU_INDEX=$(( (GPU_INDEX + 1) % ${#GPUS[@]} ))  # Update GPU index
    done
  done
done