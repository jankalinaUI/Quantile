#EVALUATION FUNCTIONS FOR Neural Networks models

import numpy as np

def mean_squared_error(y_true,y_pred):
	#residuals
	res = y_true - y_pred
	return np.mean(np.square(res))

def trimmed_mean_squared_error(y_true,y_pred, alpha = 0.75):
    h = np.int64(np.floor(alpha * y_true.shape[0]))
    r = np.square(y_true - y_pred)
    r = np.sort(r, axis = 0)
    r = r[:h]
    return np.mean(r)
