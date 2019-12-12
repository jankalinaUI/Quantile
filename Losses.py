# losses
import tensorflow as tf

import numpy as np
import scipy.special as sc


def psi(t, tt, s=None, tau=None):
    if tt == 'basic':

        t = tf.cast(t, dtype=tf.float32)
        return 1 - t

    elif tt == 'exp':

        s = 10 if s is None else s
        t = tf.cast(t, dtype=tf.float32)
        c = (1 + tf.exp(-s / 2)) / (1 + tf.exp(s * (t - 1 / 2)))
        print(c.dtype)
        return c

    elif tt == 'ind':

        t = tf.cast(t, dtype=tf.float32)
        tau = tf.constant(0.75, dtype=tf.float32) if tau is None else tau
        ones = tf.ones(tf.convert_to_tensor(t.shape[0]), dtype=tf.float32)
        zeros = tf.zeros(tf.convert_to_tensor(t.shape[0]), dtype=tf.float32)
        bl = tf.where(t < tau, ones, zeros)
        res = (1 - t / tau) * bl
        res = tf.cast(res, dtype=tf.float32)
        return res

    elif tt == 'erf':

        t = tf.cast(t, dtype=tf.float32)
        return 1 - tf.erf(t)

    elif tt == 'exp_ind':

        tau = tf.constant(0.75, dtype=tf.float32) if tau is None else tau
        s = 10 if s is None else s
        t = tf.cast(t, dtype=tf.float32)
        bl = tf.where(t < tau, ones, zeros)
        c = (1 + tf.exp(-s / 2)) / (1 + tf.exp(s * (t - tau))) * bl
        print(c.dtype)
        return c


def quantile_nonlinear(thau):
    def quantile_loss(y_true, y_pred):
        x = y_true - y_pred
        # pretoze sa bude variac tensor, toto je postup pri kerase

        return tf.reduce_sum(tf.maximum(thau * x, (thau - 1) * x))

    return quantile_loss


def least_weighted_square(tt, tau=None, s=None):
    def lws(y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float64)
        y_true = tf.cast(y_true, dtype=tf.float64)
        r = tf.subtract(y_true, y_pred)
        r = tf.square(r)
        r = tf.sort(r, axis=0)
        shape = tf.shape(r)
        c = shape.get_shape().as_list()[0]
        range = tf.range(c)
        range = range + 1
        # quite unnecessary, but for logic from R is ok for me :D 
        arr = (range - 1) / shape
        r = tf.cast(r, dtype=tf.float32)

        return tf.reduce_sum(psi(arr, tt=tt) * r)

    return lws
