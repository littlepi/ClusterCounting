#!/usr/bin/env python
# coding: utf-8

# # Train
# 
# ## Dataset preparation
# 
# Import the dataset

# In[1]:


import pandas as pd

dataset = pd.read_csv('dataset_train.txt', skipinitialspace=True)
dataset


# Split to train and validate dataset

# In[2]:


from sklearn.model_selection import train_test_split
import numpy as np

ntime = 15
ndim = 1

dataset_feature = dataset.loc[:, 'Time0':'Time%d' % (ntime-1)].values
dataset_label = dataset.loc[:, 'ID'].values

# balancing the sample
num_sig = dataset_label[dataset_label == 1].shape[0]
num_bkg = dataset_label[dataset_label == 0].shape[0]
print('number of signal = {}, number of background = {}'.format(num_sig, num_bkg))

bkg_idx = np.random.choice(np.where(dataset_label == 0)[0], num_sig, replace=False)
sig_idx = np.where(dataset_label == 1)[0]
tot_idx = np.concatenate([sig_idx, bkg_idx])
dataset_feature = dataset_feature[tot_idx]
dataset_label = dataset_label[tot_idx]

# train/validate split
dataset_feature = dataset_feature.reshape(dataset_feature.shape[0], ntime, ndim)
X_train, X_val, y_train, y_val = train_test_split(dataset_feature, dataset_label, test_size=0.33, random_state=10)


print(X_train.shape)
print(X_val.shape)


# ## Model and train
# 
# Network structure: LSTM + Dense

# In[3]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape, Flatten

# Generate model

# Define model
model = Sequential()
#model.add(Reshape((ntime, ndim), input_shape = (ntime*ndim, )))
model.add(LSTM(32, return_sequences = True, input_shape = (ntime, ndim, )))
#model.add(LSTM(32, return_sequences = True))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Set loss and optimizer
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['acc'])

# Store model to file
model.summary()


# Train the model

# In[4]:


history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, validation_data=(X_val, y_val))
history_dict = history.history
model.save('peak_classification.h5')


# ## Plots for the training

# In[5]:


import matplotlib.pyplot as plt

loss_values = history_dict['loss'][0:]
val_loss_values = history_dict['val_loss'][0:]

epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[6]:


acc_values = history_dict['acc'][0:]
val_acc_values = history_dict['val_acc'][0:]

epochs = range(1, len(acc_values)+1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

