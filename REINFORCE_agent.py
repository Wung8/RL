import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch
import torch.optim as optim

import numpy as np, scipy
import random, time, math


class REINFORCE_agent():

    def __init__(self, input_space, action_space, model=None):

        if model == None:
            self.model = nn.Sequential(
                          nn.Linear(input_space,16),
                          nn.Mish(),
                          nn.Linear(16,16),
                          nn.Mish(),
                          nn.Linear(16,action_space),
                          nn.Softmax()
                        )
        else: self.model = model
            
        self.action_space = [i for i in range(action_space)]

    def get_action(self, state, valid_actions):
        action_probabilities = self.model(self.conv(state)).detach().tolist()[0]
        action_probabilities = np.multiply(action_probabilities, valid_actions)
        action = np.random.choice(self.action_space, p=np.divide(action_probabilities,sum(action_probabilities)))
        return action, action_probabilities

    def learn(self, state, action, prob, r, scale):
        output = self.model(self.conv(state))
        grad = [0 for i in range(len(self.action_space))]
        grad[action] = -r / max(prob,0.05) * scale
        grad = torch.tensor([grad], dtype=torch.float32)
        output.backward(grad)

    def set_conv(self, conv):
        self.conv = conv

    def get_model(self):
        return self.model


'''
required functions in env:
    - resetEnv(), returns [state, valid_actions]
    - nextFrame(action), returns [next_state, r, valid_actions ,done]
    - convState(state), return [converted_state]

'''

class REINFORCE_trainer():

    def __init__(self, env, agent, optimizer=None, lr = .005,
                 batch_size = 50, discount = .97):
        self.batch_size = batch_size
        self.discount = discount
        self.agent = agent
        self.env = env
        self.agent.set_conv(self.env.convState)
        
        if optimizer == None:            
            self.opt = optim.RMSprop(self.agent.get_model().parameters(), lr=lr, weight_decay=1e-5)
        else: self.opt = optimizer

    def train(self, epochs, ep_len, verbose=True):
        for batch in range(epochs//self.batch_size):
            self.opt.zero_grad()
            # [ state, action, action_probability, r ]
            hist = [[],[],[],[]]
            # play episode
            for ep in range(self.batch_size):
                [x.extend(y) for x,y in zip(hist,self.run_episode(ep_len))]
                if ep/self.batch_size//.1 > (ep-1)/self.batch_size//.1: print('#',end='')

            # batch normalization
            lst = np.array(hist[3])
            lst = (lst-np.mean(lst)) / (np.std(lst) + 1e-10)
            hist[3] = list(lst)

            # backprop
            for info in list(zip(*hist)):
                state, action, action_probability, r = info
                self.agent.learn(state, action, action_probability, r, scale=1/len(hist[0]))
            self.opt.step()
                 
            if verbose: print(f" test score: {self.test(ep_len,display=False)}")

    def run_episode(self, ep_len):
        # [ state, action, action_probability, r ]
        hist = [[],[],[],[]]
        state, valid_actions = self.env.resetEnv()
        done = False
        for step in range(ep_len):
            action, action_probabilities = self.agent.get_action(state, valid_actions)
            next_state, r, valid_actions, done = self.env.nextFrame(action)
            [x.append(y) for x,y in zip(hist,[state, action, action_probabilities[action], r])]
            if done: break
            state = next_state

        # discount rewards
        r = hist[3]
        hist[3] = scipy.signal.lfilter([1], [1, -self.discount], x=r[::-1])[::-1]

        return hist

    def test(self, ep_len, display):
        done = False
        total_r = 0
        state, valid_actions = self.env.resetEnv()
        for step in range(ep_len):
            action, action_probabilities = self.agent.get_action(state, valid_actions)
            state, r, valid_actions, done = self.env.nextFrame(action,display=display)
            total_r += r
            if done: break
        return total_r
        
        
        
            

        
        
        

    
        
        
        

