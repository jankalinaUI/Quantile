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
from rbf import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import scipy as sc
from keras.wrappers.scikit_learn import KerasRegressor
from scipy import stats
import sklearn as sk
import pandas as p
from sklearn.model_selection import KFold
from Evaluation import mean_squared_error,trimmed_mean_squares

class QuantileNetwork:
	
  def __init__(self,thau,betas, units, input_shape,x):
		
    self.thau = thau
    self.betas = betas
    self.units = units
    self.shape = input_shape
    self.x = x
		
  #MLP_model
  def MLP_model(self,input_shape, loss):
        
        inputs = Input(shape = (input_shape,))
        layer = Dense(128, activation = K.sigmoid)(inputs)
        lay = Dense(64,activation = K.sigmoid)(layer)
        out = Dense(1)(lay)              

        model = Model(inputs = inputs , outputs = out)

        model.compile(loss = loss, optimizer = RMSprop())
       
        return model

  #RBF model		
  def RBF_model(self,x, input_shape, units, betas, loss):
    inputs = Input(shape = (input_shape,))
    rbflayer = RBFLayer(output_dim = units,
                        betas=betas,
                        initializer = InitCentersRandom(x))
    rbf = rbflayer(inputs)
    out = Dense(1)(rbf)
      
    model = Model(inputs = inputs , outputs = out)
    model.compile(loss = loss,
    optimizer = RMSprop())
        
    return model
		
  def upper_mlp(self):
  
    thau_upper = self.thau
  
    def quantile_nonlinear(y_true,y_pred):
  
      x = y_true - y_pred
      #pretoze sa bude variac tensor, toto je postup pri kerase
  
      return K.maximum(thau_upper * x,(thau_upper - 1) * x)
	
    model = self.MLP_model(input_shape = self.shape,loss = quantile_nonlinear)

    return model
	
  def lower_mlp(self):
	
    thau_lower = 1 - self.thau
  
    def quantile_nonlinear(y_true,y_pred):
  
      x = y_true - y_pred
      #pretoze sa bude variac tensor, toto je postup pri kerase
  
      return K.maximum(thau_lower * x,(thau_lower - 1) * x)
	
    model = self.MLP_model(input_shape = self.shape,loss = quantile_nonlinear)

    return model

  def upper_rbf(self):
  
    thau_upper = self.thau
  
    def quantile_nonlinear(y_true,y_pred):
  
      x = y_true - y_pred
      #pretoze sa bude variac tensor, toto je postup pri kerase
  
      return K.maximum(thau_upper * x,(thau_upper - 1) * x)
	
    model = self.RBF_model(x = self.x, input_shape = self.shape,betas = self.betas, units = self.units, loss = quantile_nonlinear)

    return model
	
  def lower_rbf(self):
	
    thau_lower = 1 - self.thau
  
    def quantile_nonlinear(y_true,y_pred):
  
      x = y_true - y_pred
      #pretoze sa bude variac tensor, toto je postup pri kerase
  
      return K.maximum(thau_lower * x,(thau_lower - 1) * x)
	
    model = self.RBF_model(x = self.x, input_shape = self.shape,betas = self.betas, units = self.units, loss = quantile_nonlinear)

    return model
  
  def evaluate(self,func,y_true,y_pred):
    
    if func == 'mean_squared_error':
      
      return mean_squared_error(y_true,y_pred)
    
    elif func == 'trimmed_mean_squared_error':
      
      return trimmed_mean_squares(y_true,y_pred,alpha = 0.75)