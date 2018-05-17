import tensorflow as tf
import tensorflow.contrib.layers as tfcl


class Critic(object):
    """ Class for the Critic-network

    Attributes
    ----------
    trainable_vars : List<Tensor>
        List of trainable variables
    action_placeholder : Tensorflow placeholder
        Input to model
    state: Tensorflow placeholder
        Input to model
    q_value: Tensor
        Output of model
    dq_da: Tensor
        Gradient of Q wrt. action
    """

    def __init__(self, state_dim, action_dim, scope="critic"):
        """Builds the grah and all necessary operations for the critic

        Parameters
        ----------
        state_dim : int
            Dimensionality of the state.
        action_dim : int
            Dimensionality of the action.
        scope : str, optional
            variable scope for the network (the default is "critic")

        """

        self._dim = state_dim
        self._state = tf.placeholder(tf.float32, [None, state_dim], "critic_state_in")
        self._action = tf.placeholder(tf.float32, [None, action_dim], "action_in")
        self._q_value = self._model(scope)
        self._dq_da = tf.gradients(self._q_value, self._action)[0]
        self._trainable_vars = tf.trainable_variables(scope)

    def _model(self, scope):
        """Adds the critic network to the graph.

        Parameters
        ----------
        scope : string
            variable scope
        Returns
        -------
        Tensor
            The output of the model, i.e. the Q-value.
        """

        with tf.variable_scope(scope):

            act = tf.nn.elu
            initializer = tfcl.variance_scaling_initializer()
            xs = tf.layers.dense(self.state, 400, act, kernel_initializer=initializer)
            xs = tf.layers.dense(xs, 300, None, kernel_initializer=initializer)

            xa = tf.layers.dense(self._action, 400, act, kernel_initializer=initializer)
            xa = tf.layers.dense(xa, 300, None, kernel_initializer=initializer)

            x = tf.nn.elu(xa + xs)

            q_value = tf.layers.dense(x, 1, None, kernel_initializer=initializer)

        return q_value

    @property
    def trainable_vars(self):
        return self._trainable_vars

    @property
    def q_value(self):
        return self._q_value

    @property
    def state(self):
        return self._state

    @property
    def action_placeholder(self):
        return self._action

    @property
    def dq_da(self):
        return self._dq_da
