import torch
import torch.utils.data as data
from torch.autograd import Variable

import numpy as np



class DataBuffer(object):
    def __init__(self, num_size_per, max_trajectory=None, store_dataset=False):
        self.data = None
        #         self.observation_dim = env.observation_space.shape[0]
        #         self.action_dim = env.action_space.shape[0]

        self.data_set = []

        self.max_trajectory = max_trajectory
        self.buffer = []
        self.store_dataset = store_dataset
        self.num_size_per = num_size_per

    def push(self, D):
        if self.store_dataset:
            self.data_set.append(D)

        self.buffer.append(D)
        if self.max_trajectory is not None:
            if len(self.buffer) > self.max_trajectory:
                del self.buffer[0]  # Delete oldest trajectory

        # self.data = np.concatenate(self.buffer, axis=0)

    def pull(self):
        return self.buffer

    def full_zero(self):

        for i in range(self.max_trajectory):
            self.push(np.zeros(self.num_size_per))



