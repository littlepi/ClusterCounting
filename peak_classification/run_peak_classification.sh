#!/bin/bash

RAW_ROOT_FILE=$1
DATASET_FILE=$2
ML_FILE=$3
OUTPUT_FILE=$4

root -q -l pre_processing.C"(\"${RAW_ROOT_FILE}\", \"${DATASET_FILE}\")"
python3 test.py --in="${DATASET_FILE}" --output="${ML_FILE}" --nsize=-1
python3 post_processing.py --dataFile="${RAW_ROOT_FILE}" --datasetFile="${DATASET_FILE}" --probFile="${ML_FILE}" --output="${OUTPUT_FILE}"
