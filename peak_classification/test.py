#!/usr/bin/env python
# coding: utf-8

# # Test
# Apply the model to the test dataset

# In[1]:


import argparse

parser = argparse.ArgumentParser(description='description')
parser.add_argument('-in', '--input', type=str, default='dataset_test.txt')
parser.add_argument('-out', '--output', type=str, default='prob_test.root')
parser.add_argument('-n', '--nsize', type=int, default=-1)
#args = parser.parse_args(args=[])
args = parser.parse_args()

filename_input = args.input
filename_output = args.output
nevt = args.nsize


# In[2]:


import pandas as pd

nevt = None if nevt < 0 else nevt
dataset = pd.read_csv(filename_input, skipinitialspace=True, nrows=nevt)
dataset


# In[3]:


ntime = 15
ndim = 1

dataset_feature = dataset.loc[:, 'Time0':'Time%d' % (ntime-1)].values
dataset_label = dataset.loc[:, 'ID'].values

dataset_feature = dataset_feature.reshape(dataset_feature.shape[0], ntime, ndim)


# In[4]:


from keras.models import load_model

model = load_model('peak_classification.h5')


# In[5]:


predicted_prob = model.predict(dataset_feature, verbose = 2)


# In[6]:


from ROOT import TFile, TTree
from array import array

fileOut = TFile.Open(filename_output, "recreate")
treeOut = TTree("tmva", "tmva")
prob_rnn = array('d', [-999])
treeOut.Branch("prob_rnn", prob_rnn, "prob_rnn/D")

for prob in predicted_prob:
    prob_rnn[0] = prob
    treeOut.Fill()

fileOut.WriteTObject(treeOut)
fileOut.Close()

