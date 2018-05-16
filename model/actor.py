import tensorflow as tf
import tensorflow.contrib.layers as tfcl
import numpy as np


class Actor(object):
    def __init__(self, state_dim, action_dim, scope="actor", action_bounds=(-2., 2.)):
        self.action_bounds = action_bounds
        self.action_dim = action_dim
        with tf.variable_scope(scope):
            self._dim = state_dim

            self._state = tf.placeholder(tf.float32, [None, state_dim], "actor_state_in")
            self._action = self._model(scope)

            self._trainable_vars = tf.trainable_variables(scope)

    def _model(self, scope):
        act = tf.nn.elu
        initializer = tfcl.variance_scaling_initializer()
        out_init = tfcl.variance_scaling_initializer(0.2)
        x = tf.layers.dense(self._state, 256, act, kernel_initializer=initializer)
        x = tf.layers.dense(x, 128, act, kernel_initializer=initializer)
        action = tf.layers.dense(x, self.action_dim, tf.nn.tanh, kernel_initializer=out_init)
        mean_act = tf.reduce_mean(self.action_bounds, keep_dims=True)
        action_amp = tf.abs(self.action_bounds[0] - self.action_bounds[0])/2
        action = (action - mean_act) * action_amp
        return action

    @property
    def trainable_vars(self):
        return self._trainable_vars

    @property
    def action(self):
        return self._action

    @property
    def state(self):
        return self._state
