import tensorlfow as tf
import tensorlfow.contrib.layers as tfcl


class Actor(object):
    def __init__(self, state_dim, scope="actor"):
        with tf.variable_scope(scope):
            self._dim = state_dim

            self._state = tf.placeholder(tf.float32, state_dim, "actor_state_in")
            self._action = self._model(scope)

            self._trainable_vars = tf.trainable_variables(scope)

    def _model(self, scope):
        act = tf.nn.elu
        initializer = tfcl.variance_scaling_initializer
        x = tf.layers.dense(self._state, 256, act, kernel_initializer=initializer)
        x = tf.layers.dense(x, 128, act, kernel_initializer=initializer)
        action = tf.layers.dense(x, 1, None, kernel_initializer=initializer)
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
