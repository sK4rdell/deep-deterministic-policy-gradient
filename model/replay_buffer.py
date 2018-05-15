from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size=1.5*10**5):

        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = deque(maxlen=int(buffer_size))
        self.a_dim, self.obs_dim = 0, 0

    def add_to_replay(self, transition):
        self._buffer.append(transition)
        if self.a_dim == 0:
            self.a_dim = transition.action.shape()
            self.obs_dim = transition.state.shape()

    @property
    def buffer_size(self):
        return len(self._buffer)

    def sample_batch(self, batch_size=256):

        if batch_size > self.buffer_size:
            num_samples = self.buffer_size
        else:
            num_samples = batch_size

        a = np.zeros(shape=[num_samples, self.a_dim])
        r = np.zeros(shape=[num_samples, 1])
        t = np.zeros(shape=[num_samples, 1])
        s1 = np.zeros(shape=[num_samples, self._obs_dim])
        s2 = np.zeros(shape=[num_samples, self._obs_dim])

        batch = random.sample(population=self._buffer, k=num_samples)

        for i in range(num_samples):
            s1[i] = batch[i].obs1
            a[i] = batch[i].action
            r[i] = batch[i].reward
            t[i] = batch[i].terminal
            s2[i] = batch[i].obs2

        return s1, a, r, t, s2
