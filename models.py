# import all the library that we need
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




# models: CNN, LSTM, Autoencoder
class CNNModel(tf.keras.Model):
    def __init__(self, cnn_filters, cnn_kernel_size, lb = None, lw = None):
        super(CNNModel, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu', use_bias=False),
            tf.keras.layers.MaxPooling1D(pool_size=4, strides=1, padding='same'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=cnn_kernel_size, activation='relu', use_bias=False),
            tf.keras.layers.MaxPooling1D(pool_size=4, strides=1, padding='same'),
            tf.keras.layers.Conv1D(filters=4, kernel_size=lb - 2 * (cnn_kernel_size - 1) - (lw - 1), activation=None, use_bias=False),
            tf.keras.layers.Dense(units=1)
        ])

    def call(self, inputs):
        return self.model(inputs)
    
    
    
class RNNModel(tf.keras.Model):
    def __init__(self, rnn_units, dropout_rate = 0.2):
        super(RNNModel, self).__init__()

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=rnn_units, return_sequences=True, activation='relu', use_bias=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=rnn_units, return_sequences=False, activation='relu', use_bias=True),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])
    def call(self, inputs):
        return self.model(inputs)


class AutoencoderModel(tf.keras.models.Model):
  def __init__(self, num_timesteps, num_inputs, num_hidden, kernel_size, pooling):
    super(AutoencoderModel, self).__init__()
    self.num = num_timesteps
    self.lb = kernel_size
    self.pooling =pooling

    encoder_input = tf.keras.Input(shape=(num_timesteps, num_inputs), name="input")
    x = tf.keras.layers.Conv1D(filters=num_hidden, kernel_size=kernel_size, activation=None, use_bias=True, padding='causal')(encoder_input)
    x = tf.keras.layers.MaxPooling1D(self.pooling, strides=self.pooling, padding='same')(x)
    self.encoder = tf.keras.Model(inputs=encoder_input, outputs=x)
    decoder_input = tf.keras.Input(shape=(int(num_timesteps/self.pooling), num_hidden))
    y = tf.keras.layers.Conv1DTranspose(filters=num_inputs, kernel_size=kernel_size, strides=self.pooling, activation=None, use_bias=True, padding='same')(decoder_input)
    self.decoder = tf.keras.Model(inputs=decoder_input, outputs=y)

  def call(self, input):
    u = self.encoder(input)
    decoded = self.decoder(u)
    return decoded


def create_model (model_type, cnn_filters=None, cnn_kernel_size=None, rnn_units=None, rnn_dropout_rate = None, lb = None, lw = None, num_inputs_auto = None, num_hidden_auto = None, kernel_size_auto = None, pooling_auto = None):

    if model_type == 'cnn':
        return CNNModel(cnn_filters, cnn_kernel_size, lb, lw)
    elif model_type == 'rnn':
        return RNNModel(rnn_units, rnn_dropout_rate)
    elif model_type == 'autoencoder':
        return AutoencoderModel(num_timesteps = lb, num_inputs = num_inputs_auto, num_hidden = num_hidden_auto, kernel_size = kernel_size_auto, pooling = pooling_auto)
    else:
        raise ValueError("Invalid model_type. Supported options are 'cnn', 'rnn', and 'autoencoder'.")
















