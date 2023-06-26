import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class WindowGenerator():
  def __init__(self, input_width, label_width, shift, input_columns=None, label_columns=None, all_columns=None):
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    self.train_label_indices = {name: i for i, name in enumerate(all_columns)}

    # ...and the input column indices
    self.input_columns = input_columns
    if input_columns is not None:
      self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
    self.train_input_indices = {name: i for i, name in enumerate(all_columns)}

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


class Autoencoder(tf.keras.models.Model):
  def __init__(self, num_timesteps, num_inputs, num_hidden, kernel_size, pooling):
    super(Autoencoder, self).__init__()
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

if __name__ == '__main__':
    # PREPARE DATA, selecting the data fo the analysis
    df = pd.read_excel('market_data.xlsx',sheet_name='US', engine='openpyxl') # read the data from the excel
    df = df.set_index(df['Date']) # set index for the data
    df = df.drop(columns='Date') # 
    df0 = df.copy()
    cols = ['EMP', 'PE', 'CAPE', 'DY', 'Rho', 'MG', 'RV', 'ED'] # this part could be changed, but reasons are needed
    df = df[cols]
    
    # denoising set-up
    df_n = df + 1.0 * np.random.normal(0, 1, df.shape)  # generate noise for the data and add it to the df
    cols_n = [str(i)+'_n' for i in cols]  # change the column name via adding "_n" to every column
    df_n.columns = cols_n  # the noisy data has appropriate column names.
    df = df.join(df_n)  
    n = len(df)  # n is the number of rows of the data
    df = df.dropna()  # drop any rows that contain missing values.

    # hold out test data
    n = len(df)
    df1 = df[0:int(0.8*n)] # the first 80% of the data is used as the training data set
    df2 = df[int(0.8*n):] # the rest 20% is used as the testing data set
    mm_scaler = preprocessing.StandardScaler()  #  standardizing the data by subtracting the mean and scaling to unit variance
    df1m = mm_scaler.fit_transform(df1) # it will get the mean and variance of the training data and transform it into the by the mean and unit variance
    df2m = mm_scaler.transform(df2) # use the information from df1 and do the same to df2
    train_df = pd.DataFrame(df1m, index=df1.index, columns=df1.columns) # form the training data
    test_df = pd.DataFrame(df2m, index=df2.index, columns=df2.columns)  # form the testing data

    # sliding window
    lb = 30
    pooling = 1
    # give lb to the window size.
    window = WindowGenerator(input_width=lb, label_width=lb, shift=0, input_columns=cols_n, label_columns=cols, all_columns=df.columns)
    
    td = window.make_dataset(train_df, shuffle=True)
    is_data = td.take(2)
    os_data = td.skip(2)

    # Training; Hint: play with num_hidden = 1 or 2, and kernel_size
    model = Autoencoder(num_timesteps=lb, num_inputs=8, num_hidden=3, kernel_size=12, pooling=pooling)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=50, decay_rate=0.97, staircase=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError(), ])
    model.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, mode='min')
    history = model.fit(is_data, validation_data=os_data, epochs=500, callbacks=[early_stopping])
    model.summary()

    fig, axs = plt.subplots()
    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.legend(['training loss', 'validation loss'])

    eval_train = window.make_dataset(train_df, batchsize=train_df.shape[0], shuffle=False)
    eval_test = window.make_dataset(test_df, batchsize=test_df.shape[0], shuffle=False)

    # CHECK IS and OS performance and P/L of a trading strategy
    plt.figure()
    plt.subplot(221)
    y_pred = model.predict(eval_train)
    y_true = np.concatenate([y for x, y in eval_train], axis=0)
    middle = model.encoder(y_true)
    mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
    plt.plot(train_df.index[lb-1:], y_true[:, -1, -1])
    plt.plot(train_df.index[lb-1:], y_pred[:, -1, -1], '-')
    plt.plot(train_df.index[lb - 1:], middle[:, -1, -1], '-')
    plt.title('in-sample mse =%1.2f' %mse )
    plt.legend(['y_true', 'y_pred', 'middle'])

    plt.subplot(222)
    y_mkt = df0.iloc[(lb-1):int(0.8*n),:].loc[:,'_MKT']
    # position taking: simple switch
    pos = ((y_pred[:,-1, 0] > 0) & (y_pred[:,-1, 1] > 0)) * 1
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    # check robustness to implementation delay
    pnl2 = pos[2:] * y_mkt[:-2]
    plt.plot(y_mkt.index[:-1], np.cumsum(pnl))
    plt.plot(y_mkt.index[:-2], np.cumsum(pnl2),'--')
    plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]))
    #relative performance
    tmp = pnl - y_mkt[:-1]
    sr = tmp.mean()/tmp.std() * np.sqrt(52)
    plt.title('in-sample IR = %1.2f' %sr)
    plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])

    plt.subplot(223)
    y_pred = model.predict(eval_test)
    y_true = np.concatenate([y for x, y in eval_test], axis=0)
    middle = model.encoder(y_true)
    mse = tf.reduce_mean(tf.keras.losses.MSE(y_true, y_pred))
    plt.plot(test_df.index[lb-1:], y_true[:,-1,-1])
    plt.plot(test_df.index[lb-1:], y_pred[:,-1,-1], '-')
    plt.plot(test_df.index[lb-1:], middle[:,-1,-1], '-')
    plt.title('out-of-sample mse =%1.2f' %mse )
    plt.legend(['y_true', 'y_pred', 'middle'])

    plt.subplot(224)
    y_mkt = df0.iloc[(lb-1)+int(0.8*n):,:].loc[:,'_MKT']
    # position taking: simple switch
    pos = ((y_pred[:,-1,0] > 0) & (y_pred[:,-1,1] > 0)) * 1
    pos[pos == -1] = 0
    pnl = pos[1:] * y_mkt[:-1]
    # check robustness to implementation delay
    pnl2 = pos[2:] * y_mkt[:-2]
    plt.plot(y_mkt.index[:-1], np.cumsum(pnl))
    plt.plot(y_mkt.index[:-2], np.cumsum(pnl2),'--')
    plt.plot(y_mkt.index[:-1], np.cumsum(y_mkt[:-1]))
    #relative performance
    tmp = pnl - y_mkt[:-1]
    sr = tmp.mean()/tmp.std() * np.sqrt(52)
    plt.title('out-of-sample IR = %1.2f' %sr)
    plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])
    plt.show()


