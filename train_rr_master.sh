# on each node, the script will only run once.
MAX_SEQ_LEN=$1
PER_DEV_BATCH_SIZE=$2
ACCU_STEP=$3
EPOCH=$4
QUERY_INSTRUCTION=$5 # bool
CORPUS_INSTRUCTION=$6 # bool
DEEPSPEED=$7 # ds_config.json or ds_config_warmup_decay.json or false
LR=$8
MAPPING=$9 # stream data
POOLING=${10}
ATTENTION=${11}
DATASET=${12}
CHECKPOINT_MODEL=${13}

# export NCCL_IB_QPS_PER_CONNECTION=8

echo "======== Hostname: ========="
RANK=${RANK:-0}
GPUS_PER_NODE=8
WORLD_SIZE=1
echo "IP Addr: $(hostname -I)"
echo "Hostname: $(hostname)"
echo "RANK: $RANK"
TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
echo "Time: $TIMESTR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODELIST="${SLURM_NODELIST}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=23456
rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT

echo WORLD_SIZE=$WORLD_SIZE RANK=$RANK rdzv_endpoint=$rdzv_endpoint
echo "MASTER_ADDR="$MASTER_ADDR

lr_scheduler_type="cosine_with_restarts" # no use if with deepspeed
IN_BATCH=true

LORA=false
LORA_R=32
MAX_Q_LEN=256
MAX_P_LEN=$MAX_SEQ_LEN
echo "DEEPSPEED=$DEEPSPEED"

echo "======== Hyperparameters: =========="
echo "Learning rate:  $LR"


RESULT_DIR="result/$(date "+%Y-%m-%d-%H%M%S")"
LOG_DIR="log"

MODEL_PATH="model/miniCPM-bf16"
DATASET_PATH="dataset/train"
CHECKPOINT_DIR="model"
RESULT_DIR="result/$(date "+%Y-%m-%d-%H%M%S")"
LOG_DIR="log"


if [ -n "$CHECKPOINT_MODEL" ]; then # a checkpoint model
  MODEL_PATH="$CHECKPOINT_DIR/$CHECKPOINT_MODEL"
  MODEL_REAL_NAME="ckpt-inherit-$MODEL_REAL_NAME"
  echo "from checkpoint $CHECKPOINT_MODEL, model path = $MODEL_PATH"
else # no checkpoint model
  echo "from scratch"
fi


IDENTITY="$TIMESTR-model-$MODEL_REAL_NAME-data-$DATASET-reranker-lr-$LR-bsz$PER_DEV_BATCH_SIZE-ngpus$GPUS_PER_NODE-nnodes$WORLD_SIZE-nepoch-$EPOCH-pooling-$POOLING-attention-$ATTENTION-qinstruct-$QUERY_INSTRUCTION-cinstruct-$CORPUS_INSTRUCTION"

echo "IDENTITY=$IDENTITY"

EXPORT_DIR="$CHECKPOINT_DIR/$IDENTITY"



echo "======== Arguments: =========="
echo "EXPORT_DIR: $EXPORT_DIR"

echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"

echo "Checkpoint Path: $CHECKPOINT_DIR"

# echo "Model Output Dir: $MODEL_OUTPUT_DIR"
echo "Result Dir: $RESULT_DIR"
echo "Log Dir: $LOG_DIR"

echo "===== mkdir tensorboard ========"
# mkdir "$CHECKPOINT_DIR/tensorboard"
# cd $CHECKPOINT_DIR
# echo "ls:"
# ls
# echo "------"


echo "======== Train begin: =========="

torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$rdzv_endpoint \
    src/openmatch/driver/train_rr.py \
    --overwrite_output_dir \
    --output_dir $EXPORT_DIR \
    --model_name_or_path $MODEL_PATH \
    --do_train  \
    --save_steps 100  \
    --train_path "$DATASET_PATH/$DATASET.jsonl" \
    --bf16 \
    --dtype "bfloat16" \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE  \
    --gradient_accumulation_steps $ACCU_STEP \
    --learning_rate $LR  \
    --q_max_len $MAX_Q_LEN  \
    --p_max_len $MAX_P_LEN  \
    --num_train_epochs $EPOCH  \
    --logging_dir "$LOG_DIR/$IDENTITY" \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --save_safetensors false \
    --query_instruction $QUERY_INSTRUCTION \
    --corpus_instruction $CORPUS_INSTRUCTION \
    --use_mapping_dataset $MAPPING \
    --pooling $POOLING \
    --attention $ATTENTION \
    --attn_implementation "flash_attention_2" \
    --train_n_passages 5  \
    --gradient_checkpointing true \

# negatives_x_device wouldn't cost much computation and time.

