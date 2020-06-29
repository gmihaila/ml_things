import gym
import numpy as np

class MyRLEnv(gym.Env):
    def __init__(self):
        super(MyRLEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Example when using discrete actions:
        N_DISCRETE_ACTIONS = 2
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        
        # Example for using image as input:
        HEIGHT, WIDTH, N_CHANNELS = 2, 2, 1
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    
        print('init basic')
        
    def step(self, action):
        # Execute one time step within the environment
        print('step')
    
    def reset(self):
        # Reset the state of the environment to an initial state
        print('reset')
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('render')
    
    def close(self):
        print('close')
        
env = MyRLEnv()

action = env.action_space.sample()

env.reset(), env.step(action), env.render(), env.close()
