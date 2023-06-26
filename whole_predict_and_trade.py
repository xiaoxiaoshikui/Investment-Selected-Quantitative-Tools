import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# get the window for the data.

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns=None, \
    label_columns=None, all_columns=None):  
    
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
    self.input_width = input_width # the input lenghth from the data set
    self.label_width = label_width  # the prediction part
    self.shift = shift # look forward length

    self.total_window_size = input_width + shift  # 

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]  # only the input window size
    

    self.label_start = self.total_window_size - self.label_width  # input_width + shift - label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice] # not sure about whether these two are equal or not.
    

  def split_window(self, features):
      inputs = features[:, self.input_slice, :] 
      labels = features[:, self.labels_slice, :]
      if self.input_columns is not None:
        inputs = tf.stack([inputs[:, :, self.train_input_indices[name]] for name in self.input_columns], axis=-1)
      if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.train_label_indices[name]] for name in self.label_columns], axis=-1)
      print("this is the shape of the input", inputs.shape)
      print("this is the shape of the output", labels.shape)
      return inputs, labels

  def make_dataset(self, data, shuffle = False, batchsize = 500,):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size,
                                                                sequence_stride = 1, sampling_rate = 1, shuffle = shuffle, batch_size = batchsize)


      ds = ds.map(self.split_window)
      return ds


if __name__ == '__main__':
    # PREPARE DATA
    df = pd.read_excel('market_data.xlsx',sheet_name='US', engine='openpyxl')
    df = df.set_index(df['Date'])
    df = df.drop(columns='Date')
    df = df.dropna()

    # hold out test data (df2)
    n = len(df)
    df1 = df[0:int(0.8*n)]
    df2 = df[int(0.8*n):]
    mm_scaler = preprocessing.StandardScaler()
    df1m = mm_scaler.fit_transform(df1)
    df2m = mm_scaler.transform(df2)
    train_df = pd.DataFrame(df1m, index=df1.index, columns=df1.columns)
    test_df = pd.DataFrame(df2m, index=df2.index, columns=df2.columns)

    # define sliding window
    lf = 52     # look forward
    ks = 26     # kernel size
    lw = 1      # label width
    lb = 52
    rnn_units = 64

    # look back
    window = WindowGenerator(input_width=lb, label_width=lw, shift=lf, input_columns=['RV', 'MG'], label_columns=['ED'])
    td = window.make_dataset(train_df, batchsize=150, shuffle=True)
    # cross-validation
    is_data = td.take(5)
    os_data = td.skip(5)
    

    # SET-UP AND TRAIN MODEL
    model_lstm = tf.keras.Sequential()
  

    model_lstm.add(tf.keras.layers.LSTM(units = rnn_units, return_sequences=True, activation='relu', use_bias=True))
    model_lstm.add(tf.keras.layers.Dropout(0.2))
    model_lstm.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
 
    model_lstm.add(tf.keras.layers.Dense(units=1))

    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=150, decay_rate=0.95, staircase=True)
    model_lstm.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError()])
    model_lstm.run_eagerly = False
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, mode='min')
    history = model_lstm.fit(is_data, validation_data=os_data, epochs=500, batch_size=150)#, callbacks=[early_stopping])
    model_lstm.summary()


    model_cnn = tf.keras.Sequential()
    # Version 1: Convolutional Network
    model_cnn.add(tf.keras.layers.Conv1D(filters=4, kernel_size=ks, activation='relu', use_bias=False))
    model_cnn.add(tf.keras.layers.MaxPooling1D(pool_size=4, strides=1, padding='same'))
    model_cnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=ks, activation='relu', use_bias=False))
    model_cnn.add(tf.keras.layers.MaxPooling1D(pool_size=4, strides=1, padding='same'))
    model_cnn.add(tf.keras.layers.Conv1D(filters=4, kernel_size=lb-2*(ks-1)-(lw-1), activation=None, use_bias=False))
    model_cnn.add(tf.keras.layers.Dense(units=1))



    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=150, decay_rate=0.95, staircase=True)
    model_cnn.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError()])
    model_cnn.run_eagerly = False
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, mode='min')
    history = model_cnn.fit(is_data, validation_data=os_data, epochs=500, batch_size=150)#, callbacks=[early_stopping])
    model_cnn.summary()


    models = [model_lstm, model_cnn]
    colors = ['blue', 'green']
    line_styles = ['-', '--']

    fig, axs = plt.subplots()
    for i, model in enumerate(models):
        axs.plot(history.history['loss'], color=colors[i], linestyle=line_styles[i])
        axs.plot(history.history['val_loss'], color=colors[i], linestyle=line_styles[i])
    axs.legend(['Training Loss (' + models[0].name + ')', 'Validation Loss (' + models[0].name + ')', 'Training Loss (' + models[1].name + ')', 'Validation Loss (' + models[1].name + ')'])

    eval_train = window.make_dataset(train_df, batchsize=train_df.shape[0], shuffle=False)
    eval_test = window.make_dataset(test_df, batchsize=test_df.shape[0], shuffle=False)

    plt.figure()
    plt.subplot(221)
    for i, model in enumerate(models):
        y_pred = model.predict(eval_train)
        y_true = np.concatenate([y for x, y in eval_train], axis=0)
        mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
        plt.plot(train_df.index[lb+lf-1:], y_true[:, -1, -1], color=colors[i])
        plt.plot(train_df.index[lb+lf-1:], y_pred[:, -1], linestyle=line_styles[i], color=colors[i])
    plt.title('In-Sample MSE')
    plt.legend(['True Value (' + models[0].name + ')', 'Predicted Value (' + models[0].name + ')', 'True Value (' + models[1].name + ')', 'Predicted Value (' + models[1].name + ')'])

    plt.subplot(222)
    y_mkt = df1.iloc[lb+lf-1:, :].loc[:, '_MKT']
    for i, model in enumerate(models):
        pos = np.sign(np.squeeze(model.predict(eval_train)[:, -1]))
        pos[pos == -1] = 0
        pnl = pos[1:] * y_mkt[:-1]
        pnl2 = pos[2:] * y_mkt[:-2]
        plt.plot(y_mkt.index[:-1], np.cumsum(pnl), color=colors[i])
        plt.plot(y_mkt.index[:-2], np.cumsum(pnl2), linestyle=line_styles[i], color=colors[i])
        plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]), color='black')
    plt.title('In-Sample P/L of a Trading Strategy')
    plt.legend(['P/L [t+1] (' + models[0].name + ')', 'P/L [t+2] (' + models[0].name + ')', 'P/L [t+1] (' + models[1].name + ')', 'P/L [t+2] (' + models[1].name + ')', 'Underlying'])

    plt.subplot(223)
    for i, model in enumerate(models):
        y_pred = model.predict(eval_test)
        y_true = np.concatenate([y for x, y in eval_test], axis=0)
        mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
        plt.plot(test_df.index[lb+lf-1:], y_true[:, -1, -1], color=colors[i])
        plt.plot(test_df.index[lb+lf-1:], y_pred[:, -1], linestyle=line_styles[i], color=colors[i])
    plt.title('Out-of-Sample MSE')
    plt.legend(['True Value (' + models[0].name + ')', 'Predicted Value (' + models[0].name + ')', 'True Value (' + models[1].name + ')', 'Predicted Value (' + models[1].name + ')'])

    plt.subplot(224)
    y_mkt = df2.iloc[lb+lf-1:, :].loc[:, '_MKT']
    for i, model in enumerate(models):
        pos = np.sign(np.squeeze(model.predict(eval_test)[:, -1]))
        pos[pos == -1] = 0
        pnl = pos[1:] * y_mkt[:-1]
        pnl2 = pos[2:] * y_mkt[:-2]
        plt.plot(y_mkt.index[:-1], np.cumsum(pnl), color=colors[i])
        plt.plot(y_mkt.index[:-2], np.cumsum(pnl2), linestyle=line_styles[i], color=colors[i])
        plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]), color='black')
    plt.title('Out-of-Sample P/L of a Trading Strategy')
    plt.legend(['P/L [t+1] (' + models[0].name + ')', 'P/L [t+2] (' + models[0].name + ')', 'P/L [t+1] (' + models[1].name + ')', 'P/L [t+2] (' + models[1].name + ')', 'Underlying'])

    plt.show()
