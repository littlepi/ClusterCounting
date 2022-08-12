#!/usr/bin/env python
# coding: utf-8

# # Test

# In[39]:


import numpy as np
import pandas as pd
import sys
import argparse
from array import array
import ROOT
from ROOT import TFile, TTree


parser = argparse.ArgumentParser()
parser.add_argument('--datasetFile', type=str, default='dataset_test.txt')
parser.add_argument('--output', type=str, default='cluster_counting.root')

args = parser.parse_args()
#args = parser.parse_args([])

filename = args.datasetFile
filename_output = args.output


# In[40]:


dataset = pd.read_csv(filename, sep='\\s+', header=None).values
print(dataset.shape)

#dataset = np.append(dataset, np.zeros((dataset.shape[0], 6)), axis=1)
dataset.shape


# In[41]:


dataset_feature = dataset[0:50000, 1:]
dataset_target = dataset[0:50000, 0]
print(dataset_feature.shape)

nfeature = 1
ntime = dataset_feature.shape[1]
nsample = dataset_feature.shape[0]

dataset_feature = dataset_feature.reshape(nsample, ntime, nfeature)
print(dataset_feature.shape)


# In[42]:


from keras.models import load_model

model = load_model('cluster_num_regression.h5')


# In[43]:


predicted = model.predict(dataset_feature)

predicted = predicted.reshape(nsample,)
#print(predicted)


# In[44]:


file_out = TFile(filename_output, 'recreate')
tree_out = TTree('signal', 'signal')

ncls_pred = array('i', [-1])
tree_out.Branch('ncls_pred', ncls_pred, 'ncls_pred/I')

for ncls in predicted:
    ncls_pred[0] = int(ncls)
    tree_out.Fill()

file_out.WriteTObject(tree_out)
file_out.Close()

