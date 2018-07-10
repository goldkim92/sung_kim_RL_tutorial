#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:01:01 2018

@author: jm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:42:12 2018

@author: jm
"""

#%%
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
import random
import DQN

env = gym.make('CartPole-v0')

#%%
def e_greedy(i, Qs):
    e = 1. / ((i//50) + 10)
    
    if np.random.rand(1) < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Qs)
    
    return action

def one_hot(x):
    return np.identity(16)[x:x+1]

def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)
    
    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)
        
        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(DQN.predict(next_state))
            
        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack, state])
    
    # Train our network using target and predicted Q values on each episode
    return DQN.update(x_stack, y_stack)

def bot_play(mainDQN):
    # See our trained network in action
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break
        
#%%
# Discount factor
lr = 0.1
dis = .9
num_episodes = 2000
REPLAY_MEMORY = 50000

# Input an doutput size based on the Env
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

replay_buffer = deque()

#%%
# create lists to contain total rewards and steps per episode 
rList = []

with tf.Session() as sess:
    mainDQN = DQN(sess, input_size, output_size)
    sess.run(tf.global_variables_initializer())
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        
        # The Q-Table learning algorithm
        while not done:
            # choose an action by e greedy
            Qs = mainDQN.predict(state)
            action = e_greedy(episode, Qs)
            
            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -100
            
            # Save the experience to our buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()
            
            state = next_state
            step_count += 1
            if step_count > 10000:
                break
        
        print("Episode: {}, steps: {}".format(episode, step_count))
        if step_count > 10000:
            pass
        
        # train every 10 episodes
        if episode % 10 == 1:
            for _ in range(50):
                minibatch = random.sample(replay_buffer, 10)
                loss, _ = simple_replay_train(mainDQN, minibatch)
            print("Loss: ", loss)
            
    bot_play(mainDQN)
