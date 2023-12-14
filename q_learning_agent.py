import numpy as np
import random


class QLearningAgent:

    def __init__(self, env, learning_rate=0.9, discount_rate=0.5, epsilon=0.5, decay_rate=0.005):
        # hyper parameters
        self.learning_rate = learning_rate  # alpha
        self.discount_rate = discount_rate  # gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate

        self.state = None

        self.action_space = env.action_space

        self.qtable = np.zeros((env.nb_states, env.nb_actions))

    def learn(self, num_episodes, max_steps, env):

        # We reset the environment for the first time
        observation = env.reset(first_reset=True)
        self.state = env.state_table.index(list(observation['agent']))

        for episode in range(num_episodes):

            for s in range(max_steps):

                action = self.choose_action()  # choose an action according to the explore/exploit policy

                observation, reward, done, info = env.step(action)
                new_state = env.state_table.index(list(observation['agent']))

                # Q-learning algorithm
                self.qtable[self.state, action] = self.qtable[self.state, action] + self.learning_rate * (
                        reward + self.discount_rate * np.max(self.qtable[new_state, :]) - self.qtable[self.state, action])

                self.state = new_state

                # if done, finish episode
                if done:
                    break

            # Decrease epsilon and reset the environment
            self.decrease_epsilon(episode)
            observation = env.reset()
            self.state = env.state_table.index(list(observation['agent']))

    def decrease_epsilon(self, episode):
        self.epsilon = np.exp(-self.decay_rate * episode)

    def choose_action(self):

        # exploration-exploitation tradeoff
        if random.uniform(0, 1) < self.epsilon:
            # explore
            action = self.action_space.sample()
        else:
            # exploit
            action = np.argmax(self.qtable[self.state, :])

        return action
