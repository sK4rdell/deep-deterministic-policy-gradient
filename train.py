import argparse
from collections import namedtuple
import gym
import numpy as np
from model.ddpg import DDPG

"""
This module contains the code for training the DDPG-agent.dd
"""


def main(env_name):
    """ Training loop for the DDPG model.

    Parameters
    ----------
    env_name : string
        name of the openAI gym evironment
    """

    env = gym.make(env_name)
    model = DDPG(init_std=.3, final_std=.05, action_dim=1, state_dim=3, alpha=.001, lr=1e-4)
    model.init()
    Transition = namedtuple("transition", ["obs1", "action", "reward", "terminal", "obs2"])
    i = 0
    while "it ain't over til it's over":
        terminal = False
        state = env.reset()
        ep_reward = 0
        while not terminal:
            env.render()
            a = model.action(np.reshape(state, [1, -1]))
            next_state, reward, terminal, _ = env.step(a)
            model.add_to_replay(Transition(action=np.squeeze(a),
                                           obs1=np.squeeze(state),
                                           reward=reward,
                                           terminal=False,
                                           obs2=np.squeeze(next_state)))
            state = next_state
            ep_reward += reward
            if i > 1:
                model.train()
        model.decay_noise()
        i += 1
        print("Episode ", i, " reward: ", ep_reward.squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate for model', type=float, default=1e-4)
    parser.add_argument('--gamma', help='discount factor', type=float, default=.99)
    parser.add_argument('--env', help='environment ID', default='Pendulum-v0')
    args = parser.parse_args()
    main(args.env)
