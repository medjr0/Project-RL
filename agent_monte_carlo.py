import numpy as np
import random
import matplotlib.pyplot as plt

class Agent_MC:

    def __init__(self, env, discount_rate=0.5, epsilon=0.4, decay_rate=0.005):
        # hyper parameters
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.discount_rate = discount_rate

        self.state = None

        # memory
        self.episode_memory = []

        # creation de la return table
        self.return_table = []
        for i in range(env.nb_states):
            line = []
            for j in range(env.nb_actions):
                line.append([])
            self.return_table.append(line)

        self.action_space = env.action_space

        self.qtable = np.zeros((env.nb_states, env.nb_actions))

        self.learning_rewards = [] # A list where will be saved the rewards of each training episode

    def learn(self, num_episodes, max_steps, env):

        # We reset the environment for the first time
        observation = env.reset(first_reset=True)
        self.state = env.state_table.index(list(observation['agent']))

        for episode in range(num_episodes):

            self.learning_rewards.append(0)

            for s in range(max_steps):

                action = self.choose_action()  # choose an action according to the explore/exploit policy

                observation, reward, done, info = env.step(action)

                # memorize state-action-reward
                self.episode_memory.append([self.state, action, reward])
                self.learning_rewards[episode] += reward

                self.state = env.state_table.index(list(observation['agent']))

                # if done, finish episode
                if done:
                    break

            # Decrease epsilon and reset the environment
            #self.decrease_epsilon(episode)
            observation = env.reset()
            self.state = env.state_table.index(list(observation['agent']))

            # complete the return function R(s,a)
            n = len(self.episode_memory)

            visited_states_actions = []

            for i in range(n):
                current_state, current_action, following_reward = self.episode_memory[i]

                if not visited_states_actions.__contains__([current_state, current_action]):
                    visited_states_actions.append([current_state, current_action])

                    for j in range(i+1, n):
                        following_reward += self.episode_memory[j][2]

                    self.return_table[current_state][current_action].append(following_reward)

                    self.qtable[current_state][current_action] = sum(self.return_table[current_state][current_action]) / len(self.return_table[current_state][current_action])

            self.episode_memory = []

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
