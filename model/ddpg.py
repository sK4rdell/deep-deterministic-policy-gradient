import tensorflow as tf

from model.actor import Actor
from model.critic import Critic
from model.exploration import Exploration


class DDPG(Exploration):
    def __init__(self, init_std, final_std):
        Exploration.__init__(self, init_std, final_std)
        self.sess = tf.Session()
        self._actor = Actor()
        self._critic = Critic()

    def action(self, state):
        pass

    def add_to_replay(self, state, action, next_state, reward, terminal):
        pass

    def train(self, batch_size=64):
        pass
