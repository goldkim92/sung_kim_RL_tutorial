#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:10:26 2018

@author: jm
"""
#%%
import gym
from gym.envs.registration import register
import readchar

#%%
# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# key mapping
arrow_keys = {
        '\x1b[A': UP,
        '\x1b[B': DOWN,
        '\x1b[C': RIGHT,
        '\x1b[D': LEFT        
        }

# choose an action by 'e greedy' or noise
e_grd = True
#%%
env = gym.make('FrozenLake-v0')
env.render() # Show the initial board

#%%
while True:
    # Choose an action from keyboard
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
    
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)
    
    if done:
        print("Finished with reward", reward)
        break