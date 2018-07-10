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

env = gym.make('FrozenLake-v0')

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

#%%
# choose an action by 'e greedy' or noise
e_grd = True

# Discount factor
lr = 0.1
dis = .99
num_episodes = 2000

# Input an doutput size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='input')
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32, name='target')
init = tf.random_uniform([input_size, output_size],0,0.01)
W = tf.get_variable(dtype=tf.float32, name='weight', initializer=init)

#
Qpred = tf.matmul(X,W)

#
loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

#%%
# create lists to contain total rewards and steps per episode 
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        done = False
        local_loss = []
        
        # The Q-Table learning algorithm
        while not done:
            # choose an action by e greedy
            Qs = sess.run(Qpred, feed_dict={X:one_hot(s)})
            action = e_greedy(i, Qs)
            
            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)
            
            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0,action] = reward
            else:
                # Obtain the Q_s1 values by feeding the new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(new_state)})
                # Update Q
                Qs[0,action] = reward + dis * np.max(Qs1)
            
            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})
            
            rAll += reward
            state = new_state
        
        rList.append(rAll)
    
# #%%
print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Qs Values")
print(Qs)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()

