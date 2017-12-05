from __future__ import division

from collections import deque
import numpy as np
import tensorflow as tf
import os, os.path

class learn:
    def __init__(self,outputname,possibleactions,weightactions,interval):
        self.outname = outputname
        self.possibleactions = possibleactions
        self.numactions = len(possibleactions)
        self.weightactions = weightactions

        self.maxbatchsize = 128
        self.replaymemorysize = 400000
        self.D = deque(maxlen=self.replaymemorysize)

        self.veclen = interval
        self.num_1stlayer = int(interval/2)
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.exploration = 0.1
        self.init_model()

    def init_model(self):
        self.x = tf.placeholder(tf.float32, [None,self.veclen])
        #fully connected
        W_fc1 = tf.Variable(tf.truncated_normal([self.veclen, self.num_1stlayer], stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.01,shape=[self.num_1stlayer]))
        h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)

        W_fc2 = tf.Variable(tf.truncated_normal([self.num_1stlayer, int(0.5*self.num_1stlayer)], stddev=0.01))
        b_fc2 = tf.Variable(tf.truncated_normal([int(0.5*self.num_1stlayer)], stddev=0.01))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        #output
        W_out = tf.Variable(tf.truncated_normal([int(0.5*self.num_1stlayer), self.numactions], stddev=0.01))
        b_out = tf.Variable(tf.constant(0.1,shape=[self.numactions]))
        self.y = tf.matmul(h_fc2,W_out) + b_out

        #loss
        self.y_ = tf.placeholder(tf.float32, [None, self.numactions])
        self.loss = tf.losses.huber_loss(self.y_,self.y)
        #self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def storeexperience(self, state_in, action_in, reward_in, statenext_in,terminal_in):
        self.D.append((state_in, action_in, reward_in,statenext_in,terminal_in))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        maxbatchsize = min(len(self.D), self.maxbatchsize)
        maxbatch_indexes = np.random.randint(0, len(self.D), maxbatchsize)

        for j in maxbatch_indexes:
            state_j, action_j, reward_j, state_jp1, terminal = self.D[j]
            action_j_index = self.possibleactions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else :
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.Q_values(state_jp1))

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # training
        self.sess.run(self.training, feed_dict={self.x: state_minibatch, self.y_: y_minibatch}) 

    def Q_values(self, state):
        return self.sess.run(self.y, feed_dict={self.x: [state]})[0]

    def set_epsilon(self, epsilon):
        self.exploratio = epsilon

    def set_lr(self, lr):
        self.learning_rate = lr
            
    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return np.random.choice(self.possibleactions,p=self.weightactions)
        else:
            return self.possibleactions[np.argmax(self.Q_values(state))]

    def select_action_norandom(self, state):
        return self.possibleactions[np.argmax(self.Q_values(state))]

    def loadmodel(self) :
        self.saver.restore(self.sess, self.outname)
    def savemodel(self):
        self.saver.save(self.sess, self.outname)
        print 'saved!'
