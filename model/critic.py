import tensorlfow as tf
import tensorflow.contrib.layers as tfcl


class Critic(object):
    def __init__(self, state_dim, action_dim, scope="critic"):
        self._dim = state_dim

        self._state = tf.placeholder(tf.float32, state_dim, "critic_state_in")
        self._action = tf.placeholder(tf.float32, action_dim, "action_in")
        self._q_value = self._model(scope)

        self._trainable_vars = tf.trainable_variables(scope)

    def _model(self, scope):
        with tf.variable_scope(scope):
            act = tf.nn.elu
            initializer = tfcl.variance_scaling_initializer
            x = tf.layers.dense(self.state, 256, act, kernel_initializer=initializer)
            x = tf.layers.dense(x, 128, act, kernel_initializer=initializer)

            adv_input = tf.concatenate([x, self._action], axis=1)

            adv = tf.layers.dense(adv_input, 1, None, kernel_initializer=initializer)
            state_value = tf.layers.dense(x, 1, None, kernel_initializer=initializer)

            q_value = adv + state_value
        return q_value

    @property
    def trainable_vars(self):
        return self._trainable_vars

    @property
    def action(self):
        return self._action

    @property
    def q_value(self):
        return self._q_value
