import numpy as np


class Exploration(object):
    """Class for policy explorartion noise.

    Methods
    -------
    exploration_noise
        sample from the exploration distribution.
    decsay_noise
        Decrease the standard deviation of the exploration distribution.
    """

    def __init__(self, init_std, final_std, steps_to_decay=400):
        """Sets initial parameters.

        Parameters
        ----------
        init_std : float
            Starting standard deviation of the exploration policy.
        final_std : float
            Final standard deviation of the expåloration policy.
        steps_to_decay : int, optional
            Number of decay steps from init to final (the default is 400)

        """

        self._delta = (init_std - final_std)/steps_to_decay
        self._std = init_std
        self._final_std = final_std

    def exploration_noise(self):
        """Samples from the exploraiton distribution.

        Returns
        -------
        np.array
            the sample.
        """

        return np.random.randn() * self._std

    def decay_noise(self):
        """Decrease the standard deviation of teh exploration policy.
        """

        self._std = np.maximum(self._std - self._delta, self._final_std)
