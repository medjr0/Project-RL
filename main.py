from simple_grid_world import GridWorldEnv
from q_learning_agent import QLearningAgent
from agent_monte_carlo import Agent_MC
from maze_world import MazeWorld
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # environment initialisation
    #env2 = GridWorldEnv(25)
    env2 = MazeWorld(20, 150)

    # agent initialisation
    agent = QLearningAgent(env2)
    #agent = Agent_MC(env2)

    # training variables
    num_episodes = 3500
    max_steps = 150

    # training
    agent.learn(num_episodes, max_steps, env2)

    print(f"Training completed over {num_episodes} episodes")
    print(agent.qtable)
    input("Press Enter to watch trained agent...")

    while(True):
        # watch trained agent
        observation = env2.reset()
        state = env2.state_table.index(list(observation['agent']))
        done = False
        rewards = 0

        for s in range(max_steps):

            print(f"TRAINED AGENT")
            print("Step {}".format(s + 1))
            action = np.argmax(agent.qtable[state, :])
            print("action :", action)
            observation, reward, done, info = env2.step(action)
            new_state = env2.state_table.index(list(observation['agent']))
            rewards += reward
            env2.render()
            print(f"score: {rewards}")
            if (np.array_equal(state, new_state)):
                break
            state = new_state

            if done == True:
                break
    env2.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
