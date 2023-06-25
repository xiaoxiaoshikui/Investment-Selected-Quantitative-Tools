import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class ModelPerformanceVisualizer:
    def __init__(self, model, eval_train, eval_test, train_df, test_df, df1, df2, lb, lf = 0):
        self.model = model
        self.eval_train = eval_train
        self.eval_test = eval_test
        self.train_df = train_df
        self.test_df = test_df
        self.df1 = df1
        self.df2 = df2
        self.lb = lb
        self.lf = lf
        self.y_pred = 0
        self.y_true = 0
        

    def plot_loss(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training loss', 'validation loss'])
        plt.show()

    def plot_in_sample_performance(self):
        plt.subplot(221)
        self.y_pred = self.model.predict(self.eval_train)
        self.y_true = tf.concat([y for x, y in self.eval_train], axis=0)
        mse = tf.reduce_mean(tf.keras.losses.MSE(self.y_true, self.y_pred))

        plt.plot(self.train_df.index[self.lb + self.lf - 1:], self.y_true[:, -1, -1])
        plt.plot(self.train_df.index[self.lb + self.lf - 1:], self.y_pred[:, -1], '--')
        plt.title('in-sample mse = %1.2f' % mse)
        plt.legend(['self.y_true', 'self.y_pred'])

    def plot_in_sample_trading_strategy(self):
        plt.subplot(222)
        y_mkt = self.df1.iloc[self.lb + self.lf - 1:, :].loc[:, '_MKT']
        pos = tf.sign(tf.squeeze(self.y_pred[:, -1]))
        pos = tf.where(tf.equal(pos, -1), 0, pos)

        pnl = pos[1:] * y_mkt[:-1]
        pnl2 = pos[2:] * y_mkt[:-2]

        plt.plot(y_mkt.index[:-1], tf.cumsum(pnl))
        plt.plot(y_mkt.index[:-2], tf.cumsum(pnl2), '--')
        plt.plot(y_mkt.index[:-1], tf.cumsum(y_mkt[:-1]))

        tmp = pnl - y_mkt[:-1]
        sr = np.mean(tmp.numpy()) / np.std(tmp.numpy()) * tf.sqrt(52.00)
        plt.title('in-sample IR = %1.2f' % sr)
        plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])

    def plot_out_of_sample_performance(self):
        plt.subplot(223)
        self.y_pred = self.model.predict(self.eval_test)
        self.y_true = tf.concat([y for x, y in self.eval_test], axis=0)
        mse = tf.reduce_mean(tf.keras.losses.MSE(self.y_true, self.y_pred))

        plt.plot(self.test_df.index[self.lb + self.lf - 1:], self.y_true[:, -1, -1])
        plt.plot(self.test_df.index[self.lb + self.lf - 1:], self.y_pred[:, -1], '--')
        plt.title('out-of-sample mse = %1.2f' % mse)
        plt.legend(['self.y_true', 'self.y_pred'])

    def plot_out_of_sample_trading_strategy(self):
        plt.subplot(224)
        y_mkt = self.df2.iloc[self.lb + self.lf - 1:, :].loc[:, '_MKT']
        pos = tf.sign(tf.squeeze(self.y_pred[:, -1]))
        pos = tf.where(tf.equal(pos, -1), 0, pos)

        pnl = pos[1:] * y_mkt[:-1]
        pnl2 = pos[2:] * y_mkt[:-2]

        plt.plot(y_mkt.index[:-1], tf.cumsum(pnl))
        plt.plot(y_mkt.index[:-2], tf.cumsum(pnl2), '--')
        plt.plot(y_mkt.index[:-1], tf.cumsum(y_mkt[:-1]))

        tmp = pnl - y_mkt[:-1]
        sr = np.mean(tmp.numpy()) / np.std(tmp.numpy()) * tf.sqrt(52.00)
        plt.title('out-of-sample IR = %1.2f' % sr)
        plt.legend(['pnl [t+1]', 'pnl [t+2]', 'underlying'])

    def show_plots(self):
        plt.show()