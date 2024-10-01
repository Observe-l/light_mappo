import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AsynchControl(object):
    def __init__(self, num_envs, num_agents):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        # Count the step for each agent
        self.cnt = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        # Indicate which agent is activated
        self.active = np.ones((self.num_envs, self.num_agents), dtype=np.int32)
        # Previous action & rewards
        self.p_active = np.ones((self.num_envs, self.num_agents), dtype=np.int32)
    
    def step(self, obs, actions):
        # Iterate all environment
        for e in range(self.num_envs):
            # Iterate all agents
            for a in range(self.num_agents):
                # Mark the agent as unactivate first
                self.active[e,a] = 0
                self.p_active[e,a] = 0
                # Check the activate agent
                if a in obs[e].keys():
                    self.cnt[e,a] += 1
                    self.active[e,a] = 1
                if a in actions[e].keys():
                    self.p_active[e,a] = 1

    def active_agents(self):
        activate_agent = []
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                if self.active[e,a]:
                    activate_agent.append((e, a, self.cnt[e,a]))
        return activate_agent
    
    def previous_agents(self):
        p_activate_agent = []
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                if self.p_active[e,a]:
                    p_activate_agent.append((e, a, self.cnt[e,a]))
        return p_activate_agent


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
