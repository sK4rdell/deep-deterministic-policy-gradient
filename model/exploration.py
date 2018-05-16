import numpy as np


class Exploration(object):
    def __init__(self, init_std, final_std, steps_to_decay=400):
        self._delta = (init_std - final_std)/steps_to_decay
        self._std = init_std
        self._final_std = final_std

    def exploration_noise(self):
        return np.random.randn() * self._std

    def decay_noise(self):
        self._std = np.maximum(self._std - self._delta, self._final_std)
