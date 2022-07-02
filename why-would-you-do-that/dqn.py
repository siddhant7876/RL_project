import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from collections import deque
from collections import namedtuple
import random

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class replay_mem(object):
    def __init__(self,cap):
        self.memory = deque([],maxlen=cap)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Dqn(nn.Module):
    def __init__(self,in_f,out_f):
        super(Dqn,self).__init__()
        self.layers=nn.Sequential(
        nn.Linear(in_f,128),
        nn.LeakyReLU(),
        #nn.Dropout(p=0.5),
        nn.Linear(128,64),
        nn.LeakyReLU(),
        #nn.Dropout(p=0.5),
        nn.Linear(64,16),
        nn.LeakyReLU(),
        #nn.Dropout(p=0.2),
        nn.Linear(16,out_f)
        )
    
    def forward(self,x):
        x=self.layers(x)
        return x.view(x.size(0),-1)
