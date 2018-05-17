import tensorflow as tf
import tensorflow.contrib.layers as tfcl


class Actor(object):
    """Attributes
    ----------
    trainable_vars : List<Tensor>
        List of trainable variables
    action : Tensor
        Policy-network endpoint
    state: Tensorflow placeholder
        Input to model
    """

    def __init__(self, state_dim, action_dim, scope="actor", action_bounds=(-2., 2.)):
        """Builds the policy graph

        Parameters
        ----------
        state_dim : int
            dimensionality of the state.
        action_dim : int
            dimensionality of the action.
        scope : str, optional
            variable scope for the network (the default is "actor")
        action_bounds : tuple, optional
            max and min value ofd the actions (the default is (-2., 2.))

        """

        self.action_bounds = action_bounds
        self.action_dim = action_dim
        with tf.variable_scope(scope):
            self._dim = state_dim

            self._state = tf.placeholder(tf.float32, [None, state_dim], "actor_state_in")
            self._action = self._model(scope)

            self._trainable_vars = tf.trainable_variables(scope)

    def _model(self, scope):
        """Adds the policy-network to the graph.

        Parameters
        ----------
        scope : string
            variable scope for the network
        Returns
        -------
        Tensor
            action
        """

        act = tf.nn.elu
        initializer = tfcl.variance_scaling_initializer()
        out_init = tfcl.variance_scaling_initializer(factor=.1)
        x = tf.layers.dense(self._state, 400, act, kernel_initializer=initializer)
        x = tf.layers.dense(x, 300, act, kernel_initializer=initializer)
        action = tf.layers.dense(x, self.action_dim, tf.nn.tanh, kernel_initializer=out_init)
        mean_act = sum(self.action_bounds) / len(self.action_bounds)
        action_amp = abs(self.action_bounds[1] - self.action_bounds[0])/2
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
