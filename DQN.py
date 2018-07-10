#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:35:17 2018

@author: jm
"""

import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, session, input_size, output_size, name='main'):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        
        self._build_network()
        
    def _build_network(self, h_size=10, lr=0.1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name='input_x')
            self._Y = tf.placeholder(tf.float32, [None, self.output_size], name='output')
            
            W1 = tf.get_variable('W1', [self.input_size, h_size], initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable('W2', [h_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
            
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            # Q prediction
            self._Qpred = tf.matmul(layer1, W2)
            
        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(lr).minimize(self._loss)
        
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})
        