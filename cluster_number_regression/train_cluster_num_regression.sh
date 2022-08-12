#!/bin/bash

PEAK_FINDING_FILE=$1
DATASET_FILE=$2

root -q -l pre_processing.C"(\"${PEAK_FINDING_FILE}\", \"${DATASET_FILE}\")"
python3 train.py
