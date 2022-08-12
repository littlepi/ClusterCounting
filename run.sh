#!/bin/bash

cd peak_classification

# Train the peak classification model
./train_peak_classification.sh "../sample/sample_0.root" "dataset_train.txt"

# Test the peak classification model, the output is also the input for the cluster number regression training
./run_peak_classification.sh "../sample/sample_1.root" "dataset_test.txt" "prob_test.root" "peak_finding_train.root"

# Generating file for the cluster number regression testing
./run_peak_classification.sh "../sample/sample_2.root" "dataset_test2.txt" "prob_test2.root" "peak_finding_test.root"

cd -


cd cluster_number_regression

# Train the cluster number regression model
./train_cluster_num_regression.sh "../peak_classification/peak_finding_train.root" "dataset_train.txt"

# Test the cluster number regression model
./run_cluster_num_regression.sh "../peak_classification/peak_finding_test.root" "dataset_test.txt" "cluster_counting.root"

cd -
