
import gym
import numpy as np
import time
env = gym.make('Taxi-v3')

Q = np.zeros([500,6])
rng =np.random.default_rng()
epsilon = 0.4
alpha = 0.7
gamma = 0.6
for _ in range(1000):
    observation_old, info = env.reset(return_info=True)
    
    env.render()
    
    done = False

    while(not done):
        if rng.random() < (epsilon)**(1+(_)/1000):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[observation_old])
     
        observation_new, reward, done, info = env.step(action)

        
        env.render()
        print(reward)
        

        current_value = Q[observation_old,action]
        Next_max = np.max(Q[observation_new])
        Q[observation_old,action] = current_value + alpha*(reward + gamma*Next_max - current_value)
        
        
    

env.close()