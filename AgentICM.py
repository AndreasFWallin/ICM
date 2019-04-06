import numpy as np
import torch
import torch.nn.functional as F
import random
from ICM import FeatureExtractNet, ICMNet, ICMLoss
from NN import QNetwork
from collections import namedtuple, deque


class Agent(object):
    """
    Reinforcment Learning Agent made to run on a NN based
    """

    def __init__(self, gamma, epsilon,
                 max_memory_size, start_memory_size,
                 alpha=0.00025, action_space=[0, 1, 2, 3],
                 eps_end=0.05, replace=10000, batch_size=32,
                 input_dim=(105,80), stepsize = 1e-4):

        self.gamma = gamma  # The discount rate
        self.epsilon = epsilon  # The exploration rate
        self.eps_end = eps_end  # The exploration end rate
        self.action_space = action_space  # The predefined action space
        self.steps = 0  # Step counter
        self.step_size= stepsize  # Define the decrease of epsilon
        self.learn_step_counter = 0  # How many times the agent has called the learn function. For network replacement
        self.memory = ReplayBuffer(action_space, max_memory_size, batch_size)  # Store object defined below
        self.outset = start_memory_size  # The initial size of the memory
        self.replace_target_cnt = replace
        self.feature_extractor = FeatureExtractNet(input_dim=input_dim)
        self.feature_dim = self.feature_extractor.feature_dim
        self.Q_policy = QNetwork(actions=len(action_space), input_size=self.feature_dim)
        self.Q_target = QNetwork(actions=len(action_space), input_size=self.feature_dim)
        self.ICM = ICMNet(action_space=action_space)
        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.Q_policy.to(self.dev)
        self.Q_target.to(self.dev)

    def take_action(self, obs):
        if self.eps < np.random.random():
            action = torch.argmax(self.Q_policy(obs))
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self, experiences, gamma):

        observations, actions, rewards, dones, next_observations = experiences

        with torch.no_grad():
            features = self.feature_extractor.forward(observations)
            next_features = self.feature_extractor.forward(next_observations)

        q_pred = self.Q_target(features)
        q_pred_next = self.Q_policy(next_features)
        maxA = torch.argmax(Q_pred, dim=1)

        q_target = q_pred
        q_pred[:,maxA] = rewards + self.gamma * torch.max(q_pred_next[1])

        if self.epsilon - self.step_size > self.eps_end:
            self.epsilon -= self.step_size
        else:
            self.epsilon = self.eps_end

        # ICM part
        features, pred_state, pred_action = self.ICM.forward(features, next_features, actions)




class ReplayBuffer:
    """
    A buffer to store a fixed number of tuples of previous values to be used for training
    """
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=
                                     ["state", "action", "reward", "next"])
        # self.seed = random.seed(seed)

    def add(self, state, action, reward, done, next_state):
        e = self.experience(state, action, reward, done, next_state)
        self.memory.append(e)

    def __len__(self):
        """
        return length of memory i.e. the experience replay size
        """
        return len(self.memory)

    def sample(self):
        """
        Get a mini_batch of experience tuple sampled ranomdly from the replay buffer
        :returns:  States, actions, rewards, dones, and next_states all of the same length as mini batch
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().permute(0, 3, 1,
                                                                                                             2).to(
            self.dev)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.dev)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.dev)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().permute(
            0, 3, 1, 2).to(self.dev)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.dev)

        return states, actions, rewards, dones, next_states
