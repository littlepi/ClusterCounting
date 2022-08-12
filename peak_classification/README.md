# Peak Classification

Classification model for peak detection from a waveform.

## Pre-processing
Preprocessing and convert the ROOT file to the text file.

Run:
```
root -q -l preprocessing.C"(\"../sample/sample_0.root\", \"dataset_train.txt\")"
```

## Train
Run the notebook train.ipynb

## Test
Run the notebook test.ipynb

## Post-processing
Run the notebook post_processing.ipynb

## Others
Besides the jupyter notebook files, there are also equivalent versions of python files for the test/post-processing. And also a shell script that wraps all processes for applying the trained model.
