import torch.nn.functional as F
from torch.nn import init
from torch import nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR
import numpy as np, scipy
import random, time, math
import keyboard as k

from buffers import Uniform_replay_buffer, Priority_replay_buffer


class DQN_agent():

    def __init__(self, input_space, action_space, model=None, epsilon=0.05):

        if model == None:
            self.model = nn.Sequential(
                          nn.Linear(input_space,36),
                          nn.Mish(),
                          nn.Linear(36,36),
                          nn.Mish(),
                          nn.Linear(36,action_space)
                        )
        else: self.model = model
            
        self.action_space = [i for i in range(action_space)]
        self.epsilon = epsilon

    def get_action(self, state, valid_actions):
        qvals = self.model(self.conv(state)).detach().tolist()[0]
        qvals = list(np.multiply(qvals, valid_actions))
        if random.random() < self.epsilon: action = random.choice(self.action_space)
        else: action = qvals.index(max(qvals))
        return action, qvals

    def learn(self, state, action, next_state, next_valid_actions, r, discount, scale):
        global max_td_e
        if next_state == -1: max_next_q = 0
        else:
            _, next_qvals = self.get_action(next_state, next_valid_actions)
            max_next_q = max(next_qvals)
        output = self.model(self.conv(state))
        q = output.detach().tolist()[0][action]
        td_e = r + discount * max_next_q - q
        grad = [0 for i in range(len(self.action_space))]
        grad[action] = -td_e * scale
        grad = torch.tensor([grad], dtype=torch.float32)
        output.backward(grad)
        if abs(td_e*scale) > max_td_e:
            max_td_e = abs(td_e*scale)
        return td_e

    def set_conv(self, conv):
        self.conv = conv

    def get_model(self):
        return self.model

    def get_modelparameters(self):
        return self.model.parameters()


'''
required functions in env:
    - resetEnv(), returns [state, valid_actions]
    - nextFrame(action), returns [next_state, r, valid_actions ,done]
    - convState(state), return [converted_state]

'''

class Epsilon_scheduler():

    def __init__(self, agent, min_epsilon=0.15, max_epsilon=1):
        self.agent = agent
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.step = self.default_update

    def default_update(self):
        self.agent.epsilon = max(self.agent.epsilon * 0.96, self.min_epsilon)

    def reset(self):
        self.agent.epsilon = self.max_epsilon


class DQN_trainer():

    def __init__(self, env, agent, discount = .97,
                 lr = .002, batch_size = 50, 
                 buffer_size=1000000, sample_size=50000):
        self.batch_size = batch_size
        self.discount = discount
        self.agent = agent
        self.env = env 
        self.agent.set_conv(self.env.convState)
        self.e_hist = []

        self.alpha, self.beta = 0.6, 0.4
        self.sample_size = sample_size
        self.replay_buffer = Priority_replay_buffer(buffer_size, sample_size, alpha=self.alpha)
        
        self.opt = optim.Adam(self.agent.get_modelparameters(), lr=lr, weight_decay=1e-5)
        self.epsilon_scheduler = Epsilon_scheduler(agent)
        #self.lr_scheduler = ExponentialLR(self.opt, gamma=0.98)

    def train(self, epochs, ep_len, verbose=True):
        for batch in range(epochs//self.batch_size):
            print(f'{batch}: ',end='')
            self.opt.zero_grad()
            # [ state, action, max_next_q, r, next_state, next_valid_actions ]
            hist = [[],[],[],[],[],[]]
            # play episode
            for ep in range(self.batch_size):
                [x.extend(y) for x,y in zip(hist,self.run_episode(ep_len))]
                if ep/self.batch_size//.1 > (ep-1)/self.batch_size//.1: print('#',end='')
            print()

            hist = list(zip(*hist))
            for i,data in enumerate(hist):
                state, action, qvals, r, next_state, next_valid_actions = data
                if i == len(hist)-1: next_qvals = [0]
                else: next_qvals = hist[i+1][2]
                td_e = abs(r + self.discount*max(next_qvals) - qvals[action])
                self.replay_buffer.add((state, action, r, next_state, next_valid_actions), td_e)

            # backprop
            indices, priorities = [],[]
            for i,info in enumerate(zip(*self.replay_buffer.select(self.beta))):
                data, weight, index = info
                state, action, r, next_state, next_valid_actions = data
                scale = weight# * self.sample_size
                td_e = self.agent.learn(state, action, next_state, next_valid_actions, r, self.discount, scale=scale)
                indices.append(index); priorities.append(abs(td_e))
            self.replay_buffer.priority_update(indices, priorities)
            nn.utils.clip_grad_norm_(self.agent.get_modelparameters(), 2.0)
            self.opt.step()
            self.epsilon_scheduler.step()
            self.beta = min(self.beta + (1-self.beta)/epochs, 1)
            #self.lr_scheduler.step()
                             
            if verbose:
                print(f" test score: {self.test(ep_len,display=True,verbose=True)}")
                for _ in range(9): print(f" test score: {self.test(ep_len,display=False)}")

    def run_episode(self, ep_len):
        # [ state, action, max_next_q, r, next_state, next_valid_actions ]
        hist = [[],[],[],[],[],[]]
        state, valid_actions = self.env.resetEnv()
        done = False
        for step in range(ep_len):
            action, qvals = self.agent.get_action(state, valid_actions)
            max_q = max(qvals)
            #if step != 0: hist[2][-1] = max_q
            next_state, r, next_valid_actions, done = self.env.nextFrame(action)
            [x.append(y) for x,y in zip(hist,[state, action, qvals, r, next_state, next_valid_actions])]
            if done: break
            state = next_state
            valid_actions = next_valid_actions

        return hist

    def test(self, ep_len, display, verbose=False):
        done = False
        total_r = 0
        state, valid_actions = self.env.resetEnv()
        show_rest = True
        for step in range(ep_len):
            action, qvals = self.agent.get_action(state, valid_actions)
            state, r, valid_actions, done = self.env.nextFrame(action,display=display and show_rest)
            if verbose and step%20 == 0 and step <= 200: print(qvals, action, r, state)
            total_r += r
            if done: break
            if k.is_pressed('space'): show_rest = False
        if verbose: print(qvals)
        return total_r

    def set_epsilon_scheduler(self, epsilon_scheduler):
        self.epsilon_scheduler = epsilon_scheduler
        
        
        
            

        
        
        

    
        
        
        

