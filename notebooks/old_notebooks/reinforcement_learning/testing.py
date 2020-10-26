"""Unit testing

To run unit testing:

python -m unittest test_dqn_helper_functions

"""

import unittest
import numpy as np
from dqn_helper_functions import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_type(self):
        # Make sure type erors are raised when necessary
        self.assertRaises(TypeError, ReplayBuffer, (1.0, 20, 3, True))
        self.assertRaises(TypeError, ReplayBuffer, (1, 2.0, 3, True))
        self.assertRaises(TypeError, ReplayBuffer, (1, 2, 3.0, True))
        self.assertRaises(TypeError, ReplayBuffer, (1, 2, 3, 'value'))
        
    def test_collect_experience(self):
        # test a sample of collect_experience
        max_size = 2
        input_shape = 20
        n_actions = 3
        discrete = False
        
        memory = ReplayBuffer(max_size, input_shape, n_actions, discrete)
        
        state = np.random.rand(input_shape)
        action = np.random.rand(n_actions)
        reward = np.random.randint(10)
        state_= np.random.rand(input_shape)
        done = False
        
        memory.collect_experience(state, action, reward, state_, done)
        
        self.assertEqual(memory.mem_cntr, 1)
        np.testing.assert_array_equal(memory.terminal_memory, [1.,0.])
        np.testing.assert_array_equal(memory.reward_memory, [reward, 0])
        np.testing.assert_array_equal(memory.action_memory, np.array([action, np.zeros(n_actions)], 
                                                                     dtype=np.float32))
        np.testing.assert_array_equal(memory.new_observation_memory, np.array([state_, np.zeros(input_shape)]))
        np.testing.assert_array_equal(memory.observation_memory, np.array([state, np.zeros(input_shape)]))
        
    def test_sample_buffer(self):
        # test a sample of collect_experience
        max_size = 2
        input_shape = 20
        n_actions = 3
        discrete = False
        batch_size = 2
        # create memory
        memory = ReplayBuffer(max_size, input_shape, n_actions, discrete)
        # create dummy environment
        state = np.random.rand(input_shape)
        action = np.random.rand(n_actions)
        reward = np.random.randint(10)
        state_ = np.random.rand(input_shape)
        done = False
        # perform 2 collection
        memory.collect_experience(state, action, reward, state_, done)
        memory.collect_experience(state, action, reward, state_, done)
        # get sample
        observations, actions, rewards, new_observations, terminal = memory.sample_buffer(batch_size)
        # test observations
        np.testing.assert_array_equal(observations, np.array([state, state]))
        # test observations
        np.testing.assert_array_equal(new_observations, np.array([state_, state_]))
        # test actions
        np.testing.assert_array_equal(actions, np.array([action, action], dtype=np.float32))
        # test rewards
        np.testing.assert_array_equal(rewards, np.array([reward, reward]))
        # test finished episode
        np.testing.assert_array_equal(terminal, np.array([1, 1], dtype=np.float32))
        
        
class TestDqnAgent(unittest.TestCase):
    
     def test_type(self):
        # Make sure type erors are raised when necessary
        pass
