#!/bin/bash

PEAK_FINDING_FILE=$1
DATASET_FILE=$2
OUTPUT_FILE=$3

root -q -l pre_processing.C"(\"${PEAK_FINDING_FILE}\", \"${DATASET_FILE}\")"
python3 test.py --datasetFile="${DATASET_FILE}" --output="${OUTPUT_FILE}"
