#!/bin/bash
date;hostname;pwd

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

conda deactivate
source activate py38_OpenIE

DATA_TYPE='aligned-sentence-pairs-all'
SERVER_PORT='8002'

DATA_DIR="../data/Newsela/newsela-auto/newsela-auto/all_data"

TSV_FILE="${DATA_DIR}/$DATA_TYPE.tsv"

OUTPUT_FILE="${DATA_DIR}/$DATA_TYPE.csv"
CACHE_FILE="${DATA_DIR}/$DATA_TYPE.json"
LOG_FILE="${DATA_DIR}/$DATA_TYPE-$(date).log"

T1=$(date +%s)

python NewselaAutoToKG.py --tsv-file "$TSV_FILE" --output-file "$OUTPUT_FILE" --log-file "$LOG_FILE" --server-port "$SERVER_PORT" --cache-file "$CACHE_FILE"
T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED"
