from __future__ import division

from collections import deque
import sys
import numpy as np
import tensorflow as tf
import os, os.path

class learn:
    def __init__(self,outputname,possibleactions,weightactions,interval):
        self.outname = outputname
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

        self.numfilter1 = 32
        self.numfilter2 = 16
        self.numfilter3 = 8
        self.numfilter4 = 4
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
        W_C1 = tf.nn.conv1d(x_,tf.truncated_normal(shape=[convlen1,1,self.numfilter1], stddev=0.01),int(convlen1/10),padding="VALID")
        b_C1 = tf.Variable(tf.truncated_normal([self.numfilter1], stddev=0.01))
        h_C1 = tf.nn.tanh( W_C1 + b_C1)

        #convoluion
        convlen2 = int(5*60/2)
        W_C2 = tf.nn.conv1d(x_,tf.truncated_normal(shape=[convlen2,1,self.numfilter2], stddev=0.01),int(convlen2/10),padding="VALID")
        b_C2 = tf.Variable(tf.truncated_normal([self.numfilter2], stddev=0.01))
        h_C2 = tf.nn.tanh( W_C2 + b_C2)

        convlen3 = int(1*60/2)
        W_C3 = tf.nn.conv1d(x_,tf.truncated_normal(shape=[convlen3,1,self.numfilter3], stddev=0.01),int(convlen3/10),padding="VALID")
        b_C3 = tf.Variable(tf.truncated_normal([self.numfilter3], stddev=0.01))
        h_C3 = tf.nn.tanh( W_C3 + b_C3)

        convlen4 = int(0.5*60/2)
        W_C4 = tf.nn.conv1d(x_,tf.truncated_normal(shape=[convlen4,1,self.numfilter4], stddev=0.01),2,padding="VALID")
        b_C4 = tf.Variable(tf.truncated_normal([self.numfilter4], stddev=0.01))
        h_C4 = tf.nn.tanh( W_C4 + b_C4)

        #Merge raw with C1
        shape_tmp = h_C1.shape
        secondd_1 = int(shape_tmp[1]*shape_tmp[2])
        shape_tmp = h_C4.shape
        secondd_4 = int(shape_tmp[1]*shape_tmp[2])
        reshape_tmp_1 = tf.reshape(h_C1,shape=[-1,secondd_1])
        reshape_tmp_4 = tf.reshape(h_C4,shape=[-1,secondd_4])

        reshape_tmp = tf.concat([reshape_tmp_1,reshape_tmp_4],axis=1)
        W_M1 = tf.Variable(tf.truncated_normal([int(reshape_tmp.shape[1]),1024], stddev=0.01))
        b_M1 = tf.Variable(tf.constant(0.01,shape=[1024]))
        h_M1 = tf.nn.tanh(tf.matmul(reshape_tmp, W_M1) + b_M1)

        shape_tmp = h_C2.shape
        secondd_2 = int(shape_tmp[1]*shape_tmp[2])
        shape_tmp = h_C3.shape
        secondd_3 = int(shape_tmp[1]*shape_tmp[2])
        reshape_tmp_2 = tf.reshape(h_C2,shape=[-1,secondd_2])
        reshape_tmp_3 = tf.reshape(h_C3,shape=[-1,secondd_3])
        reshape_tmp = tf.concat([reshape_tmp_1,reshape_tmp_4],axis=1)

        W_M2 = tf.Variable(tf.truncated_normal([int(reshape_tmp.shape[1]),1024], stddev=0.01))
        b_M2 = tf.Variable(tf.constant(0.01,shape=[1024]))
        h_M2 = tf.nn.tanh(tf.matmul(reshape_tmp, W_M2) + b_M2)

        reshape_tmp = tf.concat([h_M1,h_M2],axis=1)
        reshape_tmp = tf.reshape(reshape_tmp,[tf.shape(reshape_tmp)[0],int(reshape_tmp.shape[1]),1])
        convlenm = 100
        numfilterm = 64

        W_Cm = tf.nn.conv1d(reshape_tmp,tf.truncated_normal(shape=[convlenm,1,numfilterm], stddev=0.01),10,padding="VALID")
        b_Cm = tf.Variable(tf.truncated_normal([numfilterm], stddev=0.01))
        h_Cm = tf.nn.tanh( W_C4 + b_C4)

        shape_tmp = h_Cm.shape
        secondd_m = int(shape_tmp[1]*shape_tmp[2])
        reshape_tmp = tf.reshape(h_Cm,shape=[-1,secondd_m])

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
        self.loss = tf.reduce_mean(tf.square(self.y_ - self.y))

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
