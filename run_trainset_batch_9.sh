#!/bin/bash
date;hostname;pwd

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

conda deactivate
source activate py38_OpenIE

SERVER_PORT='8000'

BATCH_NUMBER='9'

SRC_FILE="../data/Document-level-text-simplification/Dataset/train_src_batch_$BATCH_NUMBER.txt"
TGT_FILE="../data/Document-level-text-simplification/Dataset/train_tgt_batch_$BATCH_NUMBER.txt"
OUTPUT_FILE="../data/Document-level-text-simplification/Dataset/train_batch_$BATCH_NUMBER.csv"
CACHE_FILE="../data/Document-level-text-simplification/Dataset/train_batch_$BATCH_NUMBER.json"
LOG_FILE="../data/Document-level-text-simplification/Dataset/train_batch_$BATCH_NUMBER-$(date).log"

T1=$(date +%s)

python DWikiToKG.py --src-file "$SRC_FILE" --tgt-file "$TGT_FILE" --output-file "$OUTPUT_FILE" --log-file "$LOG_FILE" --server-port "$SERVER_PORT" --cache-file "$CACHE_FILE"
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
