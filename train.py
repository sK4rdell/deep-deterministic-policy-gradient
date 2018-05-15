import argparse
from collections import namedtuple
import gym
import numpy as np
from model.ddpg import DDPG


def main(env_name):
    env = gym.make(env_name)
    model = DDPG(.2, .01, 1, 3, 1, .001, .99)

    Transition = namedtuple("transition", ["obs1", "action", "reward", "terminal", "obs2"])

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
                                           terminal=terminal,
                                           obs2=np.squeeze(next_state)))
            state = next_state
            ep_reward += reward
            model.train()
        print("Episode reward: ", ep_reward)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate for model', type=float, default=1e-4)
    parser.add_argument('--gamma', help='discount factor', type=float, default=.99)
    parser.add_argument('--env', help='environment ID', default='Pendulum-v0')
    args = parser.parse_args()
    main(args.env)
