import tensorflow as tf
import numpy as np
from model.actor import Actor
from model.critic import Critic
from model.exploration import Exploration


class DDPG(Exploration):
    def __init__(self, init_std, final_std, action_dim, alpha, gamma=.99):
        Exploration.__init__(self, init_std, final_std)
        self.gamma = .99
        self.sess = tf.Session()

        self._actor = Actor()

        self._critic = Critic()
        self._avg_critic = Critic()
        self.update_avg_critic = self.__avg_critic_update()

        with tf.namespace("training-placeholders"):
            self.td_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="td-target")

        with tf.namspace("loss-functions"):
            critic_loss = tf.reduce_mean(
                tf.squared_difference(self._critic.q_value, self.td_target))

        with tf.namespace("actor-grads"):
            self.action_grads = tf.placeholder(
                dtype=tf.float32, shape=[None, action_dim], name="action-grads")

            actor_grads = tf.gradients(
                self._actor.action, self._actor.trainable_vars, -self.action_grads)

        with tf.namespace("optimizer"):
            self._trainer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)

        with tf.namespace("update-ops"):
            self.update_critic = self._trainer.minimize(critic_loss)
            self.update_actor = self._trainer.apply_gradients(
                zip(actor_grads, self._actor.trainable_vars))

    def action(self, state):
        feed_dict = {self._actor.state: state}
        return self.sess.run(self._actor._action, feed_dict=feed_dict)

    def __avg_critic_update(self, alpha=0.01):
        return [self._avg_critic.trainable_vars[i].assign(
            tf.multiply(self._avg_critic.trainable_vars[i], alpha) +
            tf.multiply(self._critic.trainable_vars[i], 1. - alpha))
            for i in range(len(self._avg_critic._trainable_vars))]

    def add_to_replay(self, state, action, next_state, reward, terminal):
        pass

    def calculate_td_targets(self, rewards, q_vals, terminals):
        td_targets = np.zeros_like(q_vals)
        non_terminals = [not t for t in terminals]
        non_terminals = np.array(non_terminals)
        for i, r, q, t in enumerate(zip(rewards, q_vals, non_terminals)):
            td_targets[i] = r + self.gamma * q * t
        return td_targets

    def sample_from_replay():
        pass

    def _train_critic(self, _states, _actions, _rewards, _next_state, _terminals):
        feed_dict = {self._avg_critic.state: _next_state, self._avg_critic._action: _actions}
        avg_q = self.sess.run(self._avg_critic.q_value, feed_dict=feed_dict)

        td_targets = self.calculate_td_targets(_rewards, avg_q, _terminals)
        feed_dict = {self._critic.state: _states,
                     self._critic.actions: _actions, self.td_target: td_targets}

        self.sess.run([self.update_critic, self.update_avg_critic], feed_dict=feed_dict)

    def _train_actor(self, state):
        # get Q-values from average network
        action = self.sess.run(self._actor, feed_dict={self._actor.state: state})
        q = self.sess.run(self._critic.q_value, feed_dict={
                          self._critic.state: state, self._critic.action: action})

        dq_da = tf.gradients(q, action)
        feed_dict = {self._actor.state: state, self.action_grads: dq_da}
        self.sess.run(self.update_actor, feed_dict=feed_dict)

    def train(self, batch_size=64):

        _states, _actions, _rewards, _next_state, _terminals = self.sample_from_replay(batch_size)

        self._train_critic(_states, _actions, _rewards, _next_state, _terminals)
        self._train_actor(_states)
