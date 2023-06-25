# import all the library that we need
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# windows class for generating data, e.g., training data, validation data.
class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns=None, \
    label_columns=None, all_columns=None, train_df = None):  
    
    """_summary_
    
    input_ width: number of time steps of the data
    label_width: number of time steps of the prediction labels
    shift: number of time steps between the last input time step and 
           the first label time step, this one is not very sure
    """
    
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.train_label_indices = {name: i for i, name in enumerate(train_df.columns)}

    # ...and the input column indices
    self.input_columns = input_columns
    if input_columns is not None:
      self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
    self.train_input_indices = {name: i for i, name in enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.input_columns is not None:
        inputs = tf.stack([inputs[:, :, self.train_input_indices[name]] for name in self.input_columns], axis=-1)
      if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.train_label_indices[name]] for name in self.label_columns], axis=-1)
      return inputs, labels

  def make_dataset(self, data, shuffle = False, batchsize = 500,):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                                sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize)
      ds = ds.map(self.split_window)
      return ds




class DataProcessor:
    def __init__(self, data_file):
        self.data = pd.read_excel(data_file, sheet_name='US', engine='openpyxl')
        print(self.data.info())
        print(self.data.head())
        self.data = self.data.set_index(self.data['Date'])
        self.data = self.data.drop(columns='Date')
        self.data = self.data.dropna()

    def split_train_test(self, test_size=0.2):
        n = len(self.data)
        train_data = self.data.iloc[:int((1 - test_size) * n)]
        test_data = self.data.iloc[int((1 - test_size) * n):]
        return train_data, test_data

    def apply_standard_scaling(self, data):
        mm_scaler = preprocessing.StandardScaler()
        scaled_data = mm_scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
        return scaled_df