
# on each node, the script will only run once.
MAX_Q_LEN=$1
MAX_P_LEN=$2
PER_DEV_BATCH_SIZE=$3
POOLING=${4}
ATTENTION=${5}
SUB_DATASET=${6} # ArguAna, fiqa
GPUS_PER_NODE=${7}
CHECKPOINT_MODEL=${8}

# MASTER_PORT=23456

# 使用 IFS（内部字段分隔符）和 read 命令将字符串分割为数组
IFS=',' read -r -a SUB_DATASET_LIST <<< "$SUB_DATASET"

DATASET_PATH="dataset/testset"
CHECKPOINT_DIR="model"
RESULT_DIR="result/$(date "+%Y-%m-%d-%H%M%S")"
LOG_DIR="log"
CHECKPOINT_MODEL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_MODEL}"


echo "IP Addr: $(hostname -I)"
echo "Hostname: $(hostname)"
echo "RANK: $RANK"
echo "Master addr: $MASTER_ENDPOINT"
TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
echo "Time: $TIMESTR"
echo "WORLD_SIZE: $WORLD_SIZE"
\

# step1: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded query -> embedding.query.rank.{process_rank} (single file by default, hack is needed for multiple file)
# step2: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded corpus -> embedding.query.rank.{process_rank}.{begin_id}-{end_id} (multiple by default,hack is needed for single file)
# step3: distributed parallel retrieval on one node (shared storage is needed for multiple nodes), multiple gpu retrieve its part of query, and corpus will share, but load batches by batches (embedding.query.rank.{process_rank}) and save trec file trec.rank.{process_rank}
# step 4: master collect trec file and calculate metrics


for SUB_DATASET in "${SUB_DATASET_LIST[@]}"
do
    THIS_DATASET_PATH="$DATASET_PATH/$SUB_DATASET"
    THIS_LOG_DIR="$RESULT_DIR/$SUB_DATASET"
    QUERY_TEMPLATE="Query: <text>"
    CORPUS_TEMPLATE="<title> <text>"



mkdir -p "$THIS_LOG_DIR"

torchrun \
    --nnodes="1" \
    --nproc_per_node="8" \
        src/openmatch/driver/beir_eval_pipeline.py \
        --data_dir "$THIS_DATASET_PATH" \
        --model_name_or_path "$CHECKPOINT_MODEL_PATH" \
        --output_dir "$THIS_LOG_DIR" \
        --query_template "$QUERY_TEMPLATE" \
        --doc_template "$CORPUS_TEMPLATE" \
        --q_max_len $MAX_Q_LEN \
        --p_max_len $MAX_P_LEN  \
        --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir false \
        --max_inmem_docs 1000000 \
        --normalize true \
        --pooling "$POOLING" \
        --attention "$ATTENTION" \
        --phase "encode" \
        --attn_implementation "flash_attention_2" \
        # --attn_implementation "eager" \

torchrun \
    --nnodes="1" \
    --nproc_per_node="8" \
        src/openmatch/driver/beir_eval_pipeline.py \
        --model_name_or_path "$CHECKPOINT_MODEL_PATH" \
        --data_dir "$THIS_DATASET_PATH" \
        --output_dir "$THIS_LOG_DIR" \
        --use_gpu \
        --phase "retrieve" \
        --overwrite_output_dir

# rm  $THIS_LOG_DIR/embeddings.*

done