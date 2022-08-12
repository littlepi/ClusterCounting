#!/bin/bash

RAW_ROOT_FILE="../sample/sample_2.root"
DATASET_FILE="dataset_2.txt"
ML_FILE="prob_2.root"
OUTPUT_FILE="counting_2.root"

root -q -l pre_processing.C"(\"${RAW_ROOT_FILE}\", \"${DATASET_FILE}\")"
python3 test.py --in="${DATASET_FILE}" --output="${ML_FILE}" --nsize=-1
python3 post_processing.py --dataFile="${RAW_ROOT_FILE}" --datasetFile="${DATASET_FILE}" --probFile="${ML_FILE}" --output="${OUTPUT_FILE}"
