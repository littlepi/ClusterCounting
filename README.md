# ClusterCounting

A repository of cluster counting algorithm based on machine learning. The structure is organized as following:
- peak_classification: algorithm for finding all the peaks in a waveform
- cluster_num_regression: algorithm for determination the number of primary peaks in the previous step
- sample: MC samples

It is worth noting that the algorithm cannot directly apply to beam test data due to the reasons:
- Obvious difference exsits between data/MC. The trained model with MC samples cannot represent the characteristics of data well.
- One solution is to train the data directly. It requires good labelling for the data samples.
