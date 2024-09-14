
# on each node, the script will only run once.
MAX_Q_LEN=$1
MAX_P_LEN=$2
PER_DEV_BATCH_SIZE=$3
POOLING=${4}
ATTENTION=${5}
SUB_DATASET=${6} # ArguAna, fiqa
GPUS_PER_NODE=${7}
CHECKPOINT_MODEL=${8}
RETRIEVAL_RESULT_PATH=${9}

# MASTER_PORT=23456

# 使用 IFS（内部字段分隔符）和 read 命令将字符串分割为数组
IFS=',' read -r -a SUB_DATASET_LIST <<< "$SUB_DATASET"


DATASET_PATH="dataset/testset"
CHECKPOINT_DIR="model"
# CHECKPOINT_DIR="data/data_20240710/models"
RESULT_DIR="result/$(date "+%Y-%m-%d-%H%M%S")"
LOG_DIR="log"
CHECKPOINT_MODEL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT_MODEL}"


echo "EXPORT_DIR: $EXPORT_DIR"
# echo "Model Path: $MODEL_PATH"
echo "Model Path: $CHECKPOINT_MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Checkpoint Path: $CHECKPOINT_DIR"
# echo "Model REAL Name: $MODEL_REAL_NAME"
# echo "Model Output Dir: $MODEL_OUTPUT_DIR"
echo "Result Dir: $RESULT_DIR"
echo "Log Dir: $LOG_DIR"

for SUB_DATASET in "${SUB_DATASET_LIST[@]}"
do
    THIS_DATASET_PATH="$DATASET_PATH/$SUB_DATASET"
    THIS_LOG_DIR="$RESULT_DIR/$SUB_DATASET"
    THIS_TREC_PATH="$RETRIEVAL_RESULT_PATH/$SUB_DATASET"
    QUERY_TEMPLATE="Query: <text>"
    CORPUS_TEMPLATE="<title> <text>"


torchrun \
    --nnodes="1" \
    --nproc_per_node="8" \
        src/openmatch/driver/beir_eval_pipeline_rerank.py \
        --model_name_or_path "$CHECKPOINT_MODEL_PATH" \
        --data_dir "$THIS_DATASET_PATH" \
        --output_dir "$THIS_LOG_DIR" \
        --use_gpu \
        --phase "rerank" \
        --data_dir "$THIS_DATASET_PATH" \
        --model_name_or_path "$CHECKPOINT_MODEL_PATH" \
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
        --pooling "$POOLING" \
        --attention "$ATTENTION" \
        --attn_implementation "flash_attention_2" \
        --trec_run_path "$THIS_TREC_PATH" \
        --reranking_depth 100 \

done



