import numpy as np


class Exploration(object):
    def __init__(self, init_std, final_std):
        self._std = init_std
        self._final_std = final_std

    def exploration_noise(self):
        return np.random.randn() * self._std

    def decay_noise(self, delta):
        self._std = np.maximum(self._std - delta, self._final_std)
