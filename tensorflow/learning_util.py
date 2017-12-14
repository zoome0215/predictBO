from __future__ import division

from collections import deque
import sys
import numpy as np
import tensorflow as tf
import os, os.path

class learn:
    def __init__(self,outputname,possibleactions,weightactions,interval):
        self.outname = outputname
        self.outname_target = outputname+"_target"
        self.possibleactions = possibleactions
        self.numactions = len(possibleactions)
        self.weightactions = weightactions

        self.maxbatchsize = 32
        self.replaymemorysize = 400000
        self.D = deque(maxlen=self.replaymemorysize)

        self.veclen = interval
        self.num_1stlayer = int(interval/2)
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.exploration = 0.1

        self.init_model()

    def init_model(self):
        self.x = tf.placeholder(tf.float32, [None,self.veclen])
        x_ = tf.reshape(tensor=self.x,shape=[tf.shape(self.x)[0],self.veclen,1])

        #raw input
        W_fc1 = tf.Variable(tf.truncated_normal([self.veclen,1024], stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.01,shape=[1024]))
        h_fc1 = tf.nn.tanh(tf.matmul(self.x, W_fc1) + b_fc1)

        #convoluion
        convlen1 = int(10*60/2)
        numfilter1 = 64
        h_C1 = tf.nn.conv1d(x_,tf.truncated_normal(shape=[convlen1,1,numfilter1], stddev=0.01),5,padding="VALID")

        shape_tmp = h_C1.shape
        secondd = int(shape_tmp[1]*shape_tmp[2])
        reshape_tmp = tf.reshape(h_C1,shape=[-1,secondd,1])

        convlen2 = 128
        numfilter2 = 32
        h_C2 = tf.nn.conv1d(reshape_tmp,tf.truncated_normal(shape=[convlen2,1,numfilter2], stddev=0.01), \
                64,padding="VALID")


        shape_tmp = h_C2.shape
        secondd = int(shape_tmp[1]*shape_tmp[2])
        reshape_tmp = tf.reshape(h_C2,shape=[-1,secondd,1])

        convlen3 = 64
        numfilter3 = 16
        h_C3 = tf.nn.conv1d(reshape_tmp,tf.truncated_normal(shape=[convlen3,1,numfilter3], stddev=0.01), \
                64,padding="VALID")

        shape_tmp = h_C3.shape
        secondd = int(shape_tmp[1]*shape_tmp[2])
        reshape_tmp = tf.reshape(h_C3,shape=[-1,secondd])

        all_in = tf.concat([reshape_tmp,h_fc1],axis=1)

        W_A1 = tf.Variable(tf.truncated_normal([int(all_in.shape[1]),1024], stddev=0.01))
        b_A1 = tf.Variable(tf.constant(0.01,shape=[1024]))
        h_A1 = tf.nn.tanh(tf.matmul(all_in,W_A1) + b_A1)

        #output
        W_out = tf.Variable(tf.truncated_normal([1024, self.numactions], stddev=0.01))
        b_out = tf.Variable(tf.constant(0.01,shape=[self.numactions]))
        self.y = tf.matmul(h_A1,W_out) + b_out

        #loss
        self.y_ = tf.placeholder(tf.float32, [None, self.numactions])
        self. loss = tf.losses.huber_loss(self.y_, self.y)

        # train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.training = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

        # session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # target session
        self.sess_target = tf.Session()

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


    def savemodel(self):
        self.saver.save(self.sess, self.outname)
        print 'model saved!'

    def savetargetmodel(self):
        self.saver.save(self.sess_target, self.outname_target)
        print 'target model saved!'

    def loadmodel(self) :
        self.saver.restore(self.sess, self.outname)

    def loadtargetmodel(self) :
        self.saver.restore(self.sess_target, self.outname)
