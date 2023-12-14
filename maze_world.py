import gym
from gym import spaces
import pygame
import numpy as np


class MazeWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=10, nb_wall=15):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.nb_boxes = size*size # The number of states of the environment

        self.nb_wall = nb_wall
        self.nb_states = self.nb_boxes - self.nb_wall # The number of boxes which are not walls

        self.state_table = [[i, j] for i in range(self.size) for j in range(self.size)]
        self.wall_table = []

        # We generate the walls randomly
        for i in range(self.nb_wall):
            j = self.np_random.integers(1, len(self.state_table), size=1)[0]
            self.wall_table.append(self.state_table[j])
            self.state_table.remove(self.state_table[j])

        print(self.state_table)
        print(self.wall_table)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.nb_actions = 4
        self.action_space = spaces.Discrete(self.nb_actions)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, first_reset=False, return_info=False, options=None):

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.nb_states, size=2)

        while not self.state_table.__contains__(list(self._agent_location)):
            self._agent_location = self.np_random.integers(0, self.nb_states, size=2)

        if first_reset:
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location) or not self.state_table.__contains__(list(self._target_location)):
                self._target_location = self.np_random.integers(0, self.size, size=2)

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        if self.state_table.__contains__(list(self._agent_location+direction)):
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        # An episode is done if the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)

        if done:
            reward = 10
        else:
            reward = -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, agent_qtable=None, mode="human"):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        for i in range(self.nb_wall):
            pygame.draw.rect(
                canvas,
                (96, 96, 96),
                pygame.Rect(
                    pix_square_size * np.array(self.wall_table[i]),
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        
        # Draw the arrows to see agent's DECISION MAP (it le qtable is given)
        if not agent_qtable is None:
            for j in range(self.nb_states):
                X,Y = self.state_table[j]
                location = np.array([X, Y])
                action = np.argmax(agent_qtable[j, :])
                direction = self._action_to_direction[action]

                nX,nY = location + direction
                # draw the line of the arrow
                if not np.array_equal(location, self._target_location):
                    pygame.draw.line(
                        canvas,
                        (0,128,0),
                        (pix_square_size * (X+1/2),pix_square_size * (Y+1/2)),
                        (pix_square_size * (X+nX+1)/2,pix_square_size * (Y+nY+1)/2),
                        width = 2
                    )


        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    

    def save_map(self, agent_qtable, mode="human"):
        # Show the entire final map with arrows, instead of watching the agent taking only one path
        # Write here the function, when add it to the current last version of the world_maze on master branch
        
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        for i in range(self.nb_wall):
            pygame.draw.rect(
                canvas,
                (96, 96, 96),
                pygame.Rect(
                    pix_square_size * np.array(self.wall_table[i]),
                    (pix_square_size, pix_square_size),
                ),
            )
        
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )


        # draw the arrows to see agent's decision map
        for j in range(self.nb_states):
            X,Y = self.state_table[j]
            location = np.array([X, Y])
            action = np.argmax(agent_qtable[j, :])
            direction = self._action_to_direction[action]

            nX,nY = location + direction
            if not np.array_equal(location, self._target_location):
                pygame.draw.line(
                    canvas,
                    (0,128,0),
                    (pix_square_size * (X+1/2),pix_square_size * (Y+1/2)),
                    (pix_square_size * (X+nX+1)/2,pix_square_size * (Y+nY+1)/2),
                    width = 2
                    )
            

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        

        #save the agent's DECISION MAP
        current_time = time.strftime("%H%M%S",time.localtime())
        pygame.image.save(canvas, current_time+".png")

        return

