# Creating Quantile RBF netowrk class
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras import regularizers
from tensorflow.keras import layers
from keras.models import Sequential
from keras.engine.input_layer import Input
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.regularizers import l2
from sklearn.model_selection import GridSearchCV
# from RBF_tf import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import scipy as sc
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
import sklearn as sk
import pandas as p
from sklearn.model_selection import KFold
#from Evaluation import mean_squared_error, trimmed_mean_squared_error


# from Evaluation import mean_squared_error,trimmed_mean_squares
# from Losses import psi,quantile_nonlinear,least_weighted_square

class NeuralNetowrkTraining(QuantileNetwork):

    def __init__(self, train_x, train_y, test_x=None, test_y=None, thau=0.85):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.thau = thau

    def IQR_QRBF(self, units=40, betas=2.0, epochs=50, verbose=0, 
		 return_data=False,batch_size = 32):
        
        self.units = units
        self.betas = betas
        self.shape = self.train_x.shape[1]
        self.epochs = epochs
        self.verbose = verbose
        self.batch = batch_size

        obj = QuantileNetwork(x=self.train_x,
                              units=self.units,
                              betas=self.betas,
                              input_shape=self.shape,
                              thau=self.thau,
                              neurons1=None,
                              neurons2=None)

        upper_qrbf, lower_qrbf = obj.QRBF()

        # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        upper_qrbf.fit(self.train_x,
                       self.train_y,
                       epochs=self.epochs,
                       verbose=1,
                       batch_size=self.batch)
        # callbacks = [tensorboard_callback])

        lower_qrbf.fit(self.train_x,
                       self.train_y,
                       epochs=self.epochs,
                       verbose=1,
                       batch_size=self.batch)
        # callbacks = [tensorboard_callback])

        predsU = upper_qrbf.predict(self.train_x)
        predsL = lower_qrbf.predict(self.train_x)
        
        print('predsU:',predsU.shape)
        print('predsL:',predsL.shape)

        #indR = np.zeros(self.train_x.shape[0], dtype=np.int32)

        
        
        #for i in tf.range(self.train_x.shape[0]):
        #      indR[i] = np.where((self.train_y[i] <= predsU[i]) & 
        #      (self.train_y[i] >= predsL[i]), np.int64(i), np.inf)
        
        bl = np.arange(self.train_x.shape[0]).reshape(self.train_x.shape[0],1)
        indR = tf.where((self.train_y <= predsU) & (self.train_y >= predsL),
						 bl,-10)

        indR = indR[indR != -10].numpy()
        
        
        train_xR = self.train_x[indR, :]
        train_yR = self.train_y[indR]
        TRBF = obj.RBF_model(x=train_xR,
                     units=self.units,
                     betas=self.betas,
                     loss='mean_squared_error', input_shape=self.shape)

        TRBF.fit(train_xR, train_yR,
            epochs=self.epochs,
             verbose=1,
            batch_size=self.batch)
# callbacks = [tensorboard_callback],
# validation_data = (self.test_x,self.test_y))

        predsTRBF = TRBF.predict(self.test_x)

        if return_data == False:
            return predsTRBF
        elif return_data == True:
            return train_xR, train_yR


    def IQR_QMLP(self, epochs=50, verbose=0, return_data=False, neurons1=None, 
		 neurons2=None,batch_size=32):
          self.epochs = epochs
          self.vebose = verbose
          self.shape = self.train_x.shape[1]
          self.neurons1 = neurons1
          self.neurons2 = neurons2
          self.batch = batch_size 

          obj = QuantileNetwork(x=self.train_x,
                          units=None,
                          betas=None,
                          input_shape=self.shape,
                          thau=self.thau,
                          neurons1=self.neurons1,
                          neurons2=self.neurons2)

          upper_qmlp, lower_qmlp = self.QMLP()

          upper_qmlp.fit(self.train_x,
                   self.train_y,
                   epochs=self.epochs,
                   verbose=0,batch_size=self.batch)

          lower_qmlp.fit(self.train_x,
                   self.train_y,
                   epochs=self.epochs,
                   verbose=0,batch_size=self.batch)

          predsU = upper_qmlp.predict(self.train_x)
          predsL = lower_qmlp.predict(self.train_x)

          #indM = np.zeros(self.train_x.shape[0], dtype=np.int32)
          bl = np.arange(self.train_x.shape[0]).reshape(self.train_x.shape[0],1)
          indM = tf.where((self.train_y <= predsU) & (self.train_y >= predsL),
						 bl,-10)

          indM = indM[indM != -10].numpy()
          train_xM = self.train_x[indM, :]
          train_yM = self.train_y[indM]

          TMLP = obj.MLP_model(loss='mean_squared_error',
                         input_shape=self.shape,
                         neurons1=self.neurons1,
                         neurons2=self.neurons2)
          TMLP.fit(train_xM,
             train_yM,
             epochs=self.epochs,
             verbose=0,batch_size=self.batch)

          predsTMLP = TMLP.predict(self.test_x)

          if return_data == False:
              return predsTMLP
          elif return_data == True:
              return train_xM, train_yM


    def RBF_train(self, units=None, batch_size=32, betas=None, loss=None, epochs=100):
          self.units = units
          self.betas = betas
          self.loss = loss
          self.batch = batch_size

          obj = QuantileNetwork(x=self.train_x,
                          units=self.units,
                          betas=self.betas,
                          input_shape=self.shape,
                          thau=self.thau)

          model = RBF_model(x=self.train_x,
                      input_shape=self.train_x.shape[1],
                      betas=self.betas,
                      units=self.units,
                      loss=self.loss)

          model.fit(self.train_x,
              self.train_y,
              ecpohs=self.epochs,
              batch_size=self.batch,
              verbose=0)

          return model


    def MLP_train(self, batch_size=32, loss=None, epochs=100):

          self.loss = loss
          self.batch = batch_size

          obj = QuantileNetwork(x=self.train_x,
                          units=self.units,
                          betas=self.betas,
                          input_shape=self.shape,
                          thau=self.thau,
                          neurons1 = None,
                          neurons2 = None)

          model = obj.MLP_model(x=self.train_x,
                      input_shape=self.train_x.shape[1],
                      loss=self.loss)

          model.fit(self.train_x,
              self.train_y,
              ecpohs=self.epochs,
              batch_size=self.batch,
              verbose=0)

          return model


    def evaluate(self, func, y_true, y_pred,alpha = 0.75):
          
          if func == 'mean_squared_error':

              return mean_squared_error(y_true, y_pred)

          elif func == 'trimmed_mean_squared_error':

              return trimmed_mean_squared_error(y_true, y_pred, alpha=alpha)


    def final_df(self, dict_result, model_evaluate,file):
          
          if model_evaluate == 'RBF':
			
            metrics_trim = ['tmse_rbf', 'tmse_trbf']
            metrics_mean = ['mse_rbf', 'mse_trbf']
            d = {'TMSE': [dict_result[x] for x in metrics_trim],
				 'MSE': [dict_result[l] for l in metrics_mean]}
            df = p.DataFrame(data=d)
            df = df.rename(index={0: 'RBF', 1: 'TRBF'})
			
          elif model_evaluate == 'MLP':
            
            metrics_trim = ['tmse_mlp', 'tmse_tmlp']
            metrics_mean = ['mse_mlp', 'mse_tmlp']
            d = {'TMSE': [dict_result[x] for x in metrics_trim],
				 'MSE': [dict_result[l] for l in metrics_mean]}
            df = p.DataFrame(data=d)
            df = df.rename(index={0: 'MLP', 1: 'TMLP'})
          
          else:
            
            metrics_trim = ['tmse_rbf', 'tmse_trbf','tmse_mlp','tmse_tmlp']
            metrics_mean = ['mse_rbf', 'mse_trbf','mse_mlp','mse_tmlp']
            d = {'TMSE': [dict_result[x] for x in metrics_trim],
				 'MSE': [dict_result[l] for l in metrics_mean]}
            df = p.DataFrame(data=d)
            df = df.rename(index={0: 'RBF', 1: 'TRBF', 2: 'MLP', 3: 'TMLP'})
            
          if file is not None:
            df.to_csv('/content/' + file + '.csv', header = True)
          
          
          return df
