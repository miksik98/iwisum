import gym
import collections
import math
import random

import gym
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm


class QLearner:
    def __init__(self, epsilon, alfa, gamma, time_to_experiment):
        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]

        self.rewards = collections.defaultdict(lambda: collections.defaultdict(float))
        self.epsilon = epsilon
        self.alfa = alfa
        self.gamma = gamma
        self.time_to_experiment = time_to_experiment

    def learn(self, max_attempts, attempt):
        with open('0_{}.csv'.format(attempt), 'w+', encoding='UTF8') as f:
            for i in tqdm(range(max_attempts)):
                reward_sum = self.attempt()
                f.write(str(reward_sum) + '\n')

    def attempt(self):
        observation = self.discretise(self.environment.reset())
        done = False
        reward_sum = 0.0
        while not done:
            # self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        n_bins = (5, 6, 12)
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds[1:], self.upper_bounds[1:]])
        return tuple(map(int, est.transform([[observation[1], observation[2], observation[3]]])[0]))

    def pick_best_action(self, observation):
        best_action = self.environment.action_space.sample()
        max_reward = self.rewards[observation][best_action]
        for action, reward in self.rewards[observation].items():
            if reward > max_reward:
                max_reward = reward
                best_action = action
        return best_action

    def pick_action(self, observation):
        if random.uniform(0, 1) < self.get_epsilon():
            return self.environment.action_space.sample()
        return self.pick_best_action(observation)

    def update_knowledge(self, action, observation, new_observation, reward):
        self.rewards[observation][action] = (1 - self.get_alfa()) * self.rewards[observation][action] \
                                            + self.get_alfa() * \
                                            (reward + self.gamma *
                                             self.rewards[new_observation][self.pick_best_action(new_observation)])

    def update_knowledge_sarsa(self, action, observation, new_observation, reward):
        self.rewards[observation][action] += self.get_alfa() * \
                                             (reward + self.gamma *
                                              self.rewards[new_observation][self.pick_action(new_observation)] -
                                              self.rewards[observation][action])

    def get_epsilon(self):
        if self.attempt_no < self.time_to_experiment:
            return 1.0 - self.attempt_no / self.time_to_experiment
        else:
            return self.epsilon

    def get_alfa(self):
        if self.attempt_no < self.time_to_experiment:
            return 1.0 - self.attempt_no / self.time_to_experiment
        else:
            return self.alfa


def main():
    for i in range(5):
        learner = QLearner(gamma=0.98, epsilon=0.01, alfa=0.01, time_to_experiment=1500)
        learner.learn(3000, i)


if __name__ == '__main__':
    main()
