import tensorflow as tf
import numpy as np
from model.actor import Actor
from model.critic import Critic
from model.exploration import Exploration
from model.replay_buffer import ReplayBuffer


class DDPG(Exploration, ReplayBuffer):
    def __init__(self, init_std, final_std, action_dim, state_dim, alpha, gamma=.99, lr=1e-4):
        Exploration.__init__(self, init_std, final_std)
        ReplayBuffer.__init__(self, state_dim, action_dim)
        self.gamma = .99
        # self.buffer = ReplayBuffer()
        self.sess = tf.Session()

        self._actor = Actor(state_dim, action_dim)

        self._critic = Critic(state_dim, action_dim)
        self._avg_critic = Critic(state_dim, action_dim, "avg_critic")
        self.update_avg_critic = self.__avg_critic_update()

        with tf.name_scope("training-placeholders"):
            self.td_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="td-target")

        with tf.name_scope("loss-functions"):
            critic_loss = tf.reduce_mean(
                tf.squared_difference(self._critic.q_value, self.td_target))

        with tf.name_scope("actor-grads"):
            self.action_grads = tf.placeholder(
                dtype=tf.float32, shape=[None, action_dim], name="action-grads")

            actor_grads = tf.gradients(
                self._actor.action, self._actor.trainable_vars, -self.action_grads)

        with tf.name_scope("optimizer"):
            self._trainer = tf.train.AdamOptimizer(learning_rate=lr)

        with tf.name_scope("update-ops"):
            self.update_critic = self._trainer.minimize(critic_loss)
            self.update_actor = self._trainer.apply_gradients(
                zip(actor_grads, self._actor.trainable_vars))

        self.sess.run(tf.global_variables_initializer())

    def __avg_critic_update(self, alpha=0.01):
        return [self._avg_critic.trainable_vars[i].assign(
            tf.multiply(self._avg_critic.trainable_vars[i], alpha) +
            tf.multiply(self._critic.trainable_vars[i], 1. - alpha))
            for i in range(len(self._avg_critic._trainable_vars))]

    def calculate_td_targets(self, rewards, q_vals, terminals):
        td_targets = np.zeros_like(q_vals)
        non_terminals = [not t for t in terminals]
        non_terminals = np.array(non_terminals)
        i = 0
        for r, q, t in zip(rewards, q_vals, non_terminals):
            td_targets[i] = r + self.gamma * q * t
            i += 1
        return td_targets

    def _train_critic(self, _states, _actions, _rewards, _next_state, _terminals):
        feed_dict = {self._avg_critic.state: _next_state,
                     self._avg_critic.action_placeholder: _actions}
        avg_q = self.sess.run(self._avg_critic.q_value, feed_dict=feed_dict)

        td_targets = self.calculate_td_targets(_rewards, avg_q, _terminals)
        feed_dict = {self._critic.state: _states,
                     self._critic.action_placeholder: _actions, self.td_target: td_targets}

        self.sess.run([self.update_critic, self.update_avg_critic], feed_dict=feed_dict)

    def _train_actor(self, state):
        # get Q-values from average network
        action = self.sess.run(self._actor.action, feed_dict={self._actor.state: state})
        dq_da = self.sess.run(self._critic.dq_da, feed_dict={
            self._critic.state: state, self._critic.action_placeholder: action})[0]

        feed_dict = {self._actor.state: state, self.action_grads: dq_da}
        self.sess.run(self.update_actor, feed_dict=feed_dict)

    def action(self, state):
        feed_dict = {self._actor.state: state}
        a = self.sess.run(self._actor._action, feed_dict=feed_dict)

        a += np.random.normal()*0.1
        return a

    def train(self, batch_size=64):

        _states, _actions, _rewards, _terminals, _next_state = self.sample_batch(batch_size)

        self._train_critic(_states, _actions, _rewards, _next_state, _terminals)
        self._train_actor(_states)
