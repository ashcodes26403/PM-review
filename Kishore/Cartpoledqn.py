import numpy as np
import gym
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make("CartPole-v1")

gamma = 0.4
alpha = 0.7


class Q_Network(nn.Module):
    def __init__(self,obs_space):
        super(Q_Network,self).__init__()
        self.fc1 = nn.Linear(obs_space,10,bias = True)
        self.fc2 = nn.Linear(10,10,bias = True)
        self.fc3 = nn.Linear(10,2,bias = True)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self,x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x

class Target_Network(nn.Module):
    def __init__(self,obs_space):
        super(Target_Network,self).__init__()
        self.fc1 = nn.Linear(obs_space,10,bias = True)
        self.fc2 = nn.Linear(10,10,bias = True)
        self.fc3 = nn.Linear(10,2,bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        return x


        

class Agent(nn.Module):
    def __init__(self):
        super(Agent,self).__init__()
        self.new_states = np.zeros((100000,4))
        self.rewards = np.zeros(100000)
        self.actions = np.zeros(100000)
        self.states = np.zeros((100000,4))  
        


    def choose_action(self,state,epsilon_):
        Q = Q_Network(4)
        state = torch.from_numpy(state)
        self.Q = Q(state)
        
        if(random.random() < epsilon_):
            action = env.action_space.sample()
        else:
            self.Q = self.Q.detach().numpy()
            action = np.argmax(self.Q)
        return action 
    
    def get_state(self,state,action,t):
        
        new_state,reward,done,info = env.step(action)
        self.new_states[t] = new_state 
        self.rewards[t] = reward
        self.actions[t] = action
        self.states[t] = state
        return done
        
        

        
Qlearningagent = Agent()

#Experience replay
epsilon_ = 0.5

t = 0
while(t<100000):
    state = env.reset()
    done = False
    while(not done):
        epsilon_ = epsilon_*0.99
        action = Qlearningagent.choose_action(state,epsilon_)
        done = Qlearningagent.get_state(state,action,t)
        env.render()
        t = t + 1
        if(t==100000):
            break
        
    
    
 
tq = Target_Network(4)
Q = Q_Network(4)
#Training

optimizer = optim.SGD(Q.parameters(),lr = 0.01)

for epoch in range(100000):
    
    
    current_value = Q(torch.from_numpy(Qlearningagent.states[epoch]).float())[int(Qlearningagent.actions[epoch])]
    current_value = current_value.float()
    Q_nextmax = torch.tensor(Qlearningagent.rewards[epoch]) + gamma*torch.max(tq(torch.from_numpy(Qlearningagent.new_states[epoch]).float()))
    Q_nextmax =Q_nextmax.float()
    criterion = nn.MSELoss()
    loss = criterion(current_value,Q_nextmax)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    if (epoch+1)%100 == 0:
        print(f'epoch: {epoch+1}, loss : {loss.item()}')
    

for i in range(1000):
    state = env.reset()
    
    done = False
    while(not done):
        action = Qlearningagent.choose_action(state,epsilon_)
        new_state,reward,done,info= env.step(action)
        
        env.render()
 

