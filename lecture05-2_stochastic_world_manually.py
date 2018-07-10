#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:14:21 2018

@author: jm
"""

#%%
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

#%%
def e_greedy(i, Q):
    e = 1. / ((i//100) + 1)
    
    if np.random.rand(1) < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state,:])
    
    return action
#%%
# choose an action by 'e greedy' or noise
e_grd = True

# initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Discount factor
lr = 0.6
dis = .99
num_episodes = 2000

# create lists to contain total rewards and steps per episode 
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    
    # The Q-Table learning algorithm
    while not done:
        if e_grd == True:
            # choose an action by e greedy
            action = e_greedy(i, Q)
        else:
            # choose an action by greedily (with noise) picking from Q table
            action = np.argmax(Q[state, :] + np.random.randn(env.action_space.n) / (i+1))
        
        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)
        
        #Update Q-Table with new knowledge using decay rate
#        Q[state, action] = reward + dis * np.max(Q[new_state,:])
        Q[state, action] = (1-lr) * Q[state, action]  +lr * (reward + dis * np.max(Q[new_state,:]))
        
        rAll += reward
        state = new_state
    
    rList.append(rAll)
    
# #%%
print("Success rate: " + str(sum(rList)/num_episodes))
#print("Final Q-Table Values")
#print(Q)
#plt.bar(range(len(rList)), rList, color='blue')
#plt.show()