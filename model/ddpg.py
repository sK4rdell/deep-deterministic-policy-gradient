import tensorflow as tf

from model.actor import Actor
from model.critic import Critic
from model.exploration import Exploration


class DDPG(Exploration):
    def __init__(self, init_std, final_std, alpha):
        Exploration.__init__(self, init_std, final_std)
        self.sess = tf.Session()
        self._actor = Actor()
        self._avg_actor = Actor()
        self._update_avg_actor = self.__avg_actor_update(alpha)
        self._critic = Critic()

    def action(self, state):
        feed_dict = {self._actor.state: state}
        return self.sess.run(self._actor._action, feed_dict=feed_dict)

    def __avg_actor_update(self, alpha=0.1):
        return [self._avg_actor.trainable_vars[i].assign(
            tf.multiply(self._avg_actor.trainable_vars[i], alpha) +
            tf.multiply(self._actor.trainable_vars[i], 1. - alpha))
            for i in range(len(self._avg_actor._trainable_vars))]

    def add_to_replay(self, state, action, next_state, reward, terminal):
        pass

    def train(self, batch_size=64):

        state = 0
        _action = 0
        action = self.sess.run(self._actor, feed_dict={self._actor.state: state})
        q = self.sess.run(self._critic.q_value, feed_dict={self._critic.state: state})
        dq_da = tf.gradients(q, _action)

        pass
