import tensorflow as tf
import numpy as np
from model.actor import Actor
from model.critic import Critic
from model.exploration import Exploration
from model.replay_buffer import ReplayBuffer


class DDPG(Exploration, ReplayBuffer):
    """Implements the paper Deep Deterministic Policy Gradient algorithm
    https://arxiv.org/abs/1509.02971. Inherits from Exploraiton and ReplayBuffer.

    Methods
    -------
    train
        Performes a training step on the actor and the critic
    add_to_replay
        Add transition to replay buffer

    """

    def __init__(self, init_std,
                 final_std,
                 action_dim,
                 state_dim,
                 alpha,
                 batch_size=128,
                 gamma=.99,
                 lr=1e-4):
        """Builds up the graph and all neccesary operations for the model.

        Parameters
        ----------
        init_std : float
            initial standard deviation for the exploration noise.
        final_std : float
            Final standard deviation for the exploration noise.
        action_dim : int
            Dimensionality of the actions.
        state_dim : int
            Dimensionality of the states.
        alpha : float
            parameter for the updates of the average networks, i.e.
            avg_net = net * alpha + avg_net * (1 - alpha)
        batch_size : int, optional
            Batch size for training (the default is 128)
        gamma : float, optional
            Discount factor (the default is .99)
        lr : [type], optional
            Learning rate for the optimizers (the default is 1e-4)

        """

        Exploration.__init__(self, init_std, final_std, 1000)
        ReplayBuffer.__init__(self, state_dim, action_dim)
        self.batch_size = batch_size
        self.gamma = .99
        self.sess = tf.Session()

        self._actor = Actor(state_dim, action_dim)
        self._avg_actor = Actor(state_dim, action_dim, scope="avg_actor")

        self.update_avg_actor = self.__avg_params_update(
            self._actor.trainable_vars, self._avg_actor.trainable_vars)

        self._critic = Critic(state_dim, action_dim)
        self._avg_critic = Critic(state_dim, action_dim, scope="avg_critic")

        self.update_avg_critic = self.__avg_params_update(
            self._critic.trainable_vars, self._avg_critic.trainable_vars)

        with tf.name_scope("training-placeholders"):
            self.td_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="td-target")

        with tf.name_scope("loss-functions"):
            critic_loss = tf.reduce_mean(
                tf.squared_difference(self._critic.q_value, self.td_target))

        with tf.name_scope("actor-grads"):
            self.action_grads = tf.placeholder(
                dtype=tf.float32, shape=[None, action_dim], name="action-grads")

            actor_grads = tf.gradients(
                ys=self._actor.action, xs=self._actor.trainable_vars, grad_ys=-self.action_grads)

        with tf.name_scope("optimizers"):
            self._critic_trainer = tf.train.AdamOptimizer(learning_rate=5 * lr)
            self._actor_trainer = tf.train.AdamOptimizer(learning_rate=lr)

        with tf.name_scope("update-ops"):
            self.update_critic = self._critic_trainer.minimize(
                critic_loss, var_list=self._critic.trainable_vars)

            self.update_actor = self._actor_trainer.apply_gradients(
                grads_and_vars=zip(actor_grads, self._actor.trainable_vars))

    def __avg_params_update(self, train_vars, avg_train_vars, alpha=0.01):
        """Tensorflow op for a running average o model parameters.

        Parameters
        ----------
        train_vars : List<Tensors>
            List of main parameters.
        avg_train_vars : List<Tensor>
            List of the parameters for the running average.
        alpha : float, optional
            [description] How much of train_vars that will be added to avg_train_vars 
            (the default is 0.01)

        Returns
        -------
        Tensorflow Operation
            Operation for the runnign average
        """

        return [avg_train_vars[i].assign(tf.multiply(train_vars[i], alpha) +
                                         tf.multiply(avg_train_vars[i], 1. - alpha))
                for i in range(len(avg_train_vars))]

    def calculate_td_targets(self, rewards, q_vals, terminals):
        """Calculates the target value for the temporal difference error.

        Parameters
        ----------
        rewards : List<float>
            List of rewards.
        q_vals : List<float>
            list of Q(s_t+1, a_st+1)
        terminals : List<boolean>
            List of booleans, that are true if s_t+1 is a terminal state.
        Returns
        -------
        List<float>
            List of floats containing the td-targets.
        """

        td_targets = np.zeros_like(q_vals)
        non_terminals = [not t for t in terminals]
        non_terminals = np.array(non_terminals, dtype=np.float32)
        i = 0
        for r, q, t in zip(rewards, q_vals, non_terminals):
            td_targets[i] = r + self.gamma * q * t
            i += 1
        return td_targets

    def _train_critic(self, states, actions, rewards, next_state, terminals):
        """Performes on training step on the critic.

        Parameters
        ----------
        states : List<np.array>
            Set of starting states.
        actions : List<np.array>
            List of actions.
        rewards : List<float>
            List of rewards.
        next_state : List<np.array>
            List of ending states for hte transition.
        terminals : List<boolean>
            List of booleans, that are true for terminal states.
        """

        # get action from average policy
        feed_dict = {self._avg_actor.state: next_state}
        action = self.sess.run(self._avg_actor.action, feed_dict=feed_dict)
        # calculate Q-values from next state and action from avg-policy
        feed_dict = {self._avg_critic.state: next_state,
                     self._avg_critic.action_placeholder: action}
        self.sess.run(self._avg_critic.q_value, feed_dict=feed_dict)

        td_targets = self.calculate_td_targets(rewards, avg_q, terminals)
        feed_dict = {self._critic.state: states,
                     self._critic.action_placeholder: actions, self.td_target: td_targets}

        self.sess.run([self.update_critic, self.update_avg_critic], feed_dict=feed_dict)

    def _train_actor(self, state):
        """Performes one training step on the actor.

        Parameters
        ----------
        state : List<np.array>
            List of starting states.
        """

        # get Q-values from average network
        action = self.sess.run(self._actor.action, feed_dict={self._actor.state: state})
        dq_da = self.sess.run(self._critic.dq_da, feed_dict={
            self._critic.state: state, self._critic.action_placeholder: action})

        feed_dict = {self._actor.state: state, self.action_grads: dq_da}
        self.sess.run([self.update_actor, self.update_avg_actor], feed_dict=feed_dict)

    def action(self, state, noise=True):
        """Outputs an action from the current policy

        Parameters
        ----------
        state : np.array
            current state.
        noise : bool, optional
            Exploraution noise (the default is True)

        Returns
        -------
        np.array
            action.
        """

        feed_dict = {self._actor.state: state}
        a = self.sess.run(self._actor._action, feed_dict=feed_dict)
        if noise:
            a += self.exploration_noise()
        return a

    def train(self):
        """Performes on trainig step on the actor and the critic.
        """

        _states, _actions, _rewards, _terminals, _next_state = self.sample_batch(self.batch_size)

        self._train_critic(_states, _actions, _rewards, _next_state, _terminals)
        self._train_actor(_states)

    def init(self):
        """Initializes the graph.
        """

        self.sess.run(tf.global_variables_initializer())
