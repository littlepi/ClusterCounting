#!/bin/bash

PEAK_FINDING_FILE="../peak_classification/peak_finding_2.root"
DATASET_FILE="dataset_test.txt"
OUTPUT_FILE="cluster_counting_test.root"

root -q -l pre_processing.C"(\"${PEAK_FINDING_FILE}\", \"${DATASET_FILE}\")"
python3 test.py --datasetFile="${DATASET_FILE}" --output="${OUTPUT_FILE}"
