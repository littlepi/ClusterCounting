#!/bin/bash

RAW_ROOT_FILE=$1
DATASET_FILE=$2

root -q -l pre_processing.C"(\"${RAW_ROOT_FILE}\", \"${DATASET_FILE}\")"
python3 train.py
