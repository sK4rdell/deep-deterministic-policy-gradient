from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    """ Class for the experience replay buffer.

     Attributes
    ----------
    buffer_size : int
        number of items in the buffer.

    Methods
    -------
    sample_batch
        samples a set of tanisitions from the buffer
    add_to_replay
        Decrease the standard deviation of the exploration distribution.
    """

    def __init__(self, state_dim, action_dim, buffer_size=100000):
        """ Initiates the buffer.

        Parameters
        ----------
        state_dim : dimensionality of the state
            [description]
        action_dim : [type]
            [description]
        buffer_size : int, optional
            [description] (the default is 100000, which [default_description])

        """

        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = deque(maxlen=int(buffer_size))
        self._a_dim = action_dim
        self._obs_dim = state_dim

    def add_to_replay(self, transition):
        """Adds a transition to the buffer.

        Parameters
        ----------
        transition : named_tuple
            tuple containing abs1, action, reward, obs2, terminal
        """

        self._buffer.append(transition)

    @property
    def buffer_size(self):
        return len(self._buffer)

    def sample_batch(self, batch_size=256):
        """samples a set of transitions uniformly from the buffer.

        batch_size : int, optional
            Number of transitions to sample from the batch (the default is 256)

        Returns
        -------
        np.array, np.array, np.array, np.array, np.array 
            obersvations_0, actions, rewards, terminals, observations_1
        """

        if batch_size > self.buffer_size:
            num_samples = self.buffer_size
        else:
            num_samples = batch_size

        a = np.zeros(shape=[num_samples, self._a_dim])
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
