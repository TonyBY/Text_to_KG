#!/bin/bash
date;hostname;pwd

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

conda deactivate
source activate py38_OpenIE

DATA_TYPE='train'
SERVER_PORT='8000'

SRC_FILE="../data/Document-level-text-simplification/Dataset/$DATA_TYPE.src"
TGT_FILE="../data/Document-level-text-simplification/Dataset/$DATA_TYPE.tgt"
OUTPUT_FILE="../data/Document-level-text-simplification/Dataset/$DATA_TYPE.csv"
CACHE_FILE="../data/Document-level-text-simplification/Dataset/$DATA_TYPE.json"
LOG_FILE="../data/Document-level-text-simplification/Dataset/$DATA_TYPE-$(date).log"

T1=$(date +%s)

python DWikiToKG.py --src-file "$SRC_FILE" --tgt-file "$TGT_FILE" --output-file "$OUTPUT_FILE" --log-file "$LOG_FILE" --server-port "$SERVER_PORT" --cache-file "$CACHE_FILE"
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
