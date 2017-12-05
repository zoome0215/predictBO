#!/usr/bin/python
from __future__ import division

import numpy as np
import os, os.path
import sys
import time
from datetime import datetime
import pickle

#graphing
import matplotlib.pyplot as plt

#own python dependencies
import data_util
import learning_util

upparam = 1
downparam = -1
restparam = 0

#############################
interval = 15 # min 
betinterval = 5 # min
periodint = 2

outname = 'realuod_'+str(interval)+'_'+str(betinterval)+'_tf'

target_rate = 4 # per hour
payrate=2
initmoney = 200
bet = 20

test_year = 2017

wait = 5 #min

possibleactions = (downparam,restparam,upparam)
weightactions = np.array([0.2,0.6,0.2])

if betinterval < 10 :
    payrate = 1.85

gain = bet*(payrate-1)
loss = (periodint*bet*target_rate/60)/10

print 'tested on', datetime.today()
print 'interval of ' , interval ,' min'
print 'loss is', loss
print 'making AI...'

#Converting the intervals to array length
interval = int(interval*60/periodint)
betinterval = int(betinterval*60/periodint)
wait = int(wait*60/2)

#making AI
outdir  = '../AI/tf/'
outname = outdir+outname


learner = learning_util.learn(outname,possibleactions,weightactions,interval)
learner.loadmodel()

#load data
datanow = data_util.tradedata()

moneynow=0
numchances = 0 
numTAs = 0
for month in range(1,2):
    for i in range(0,1000):
        datanow.loaddata(test_year,month,i)
        if (not datanow.exist_data) :
            break
        jlim = datanow.size()-(interval+betinterval)
        if jlim >100 :
            j=0
            print test_year,month,i
            numchances += jlim
            while j < jlim :
                datnowall = datanow.get(j,interval+betinterval)
                state = datnowall[:interval]
                diff_io=state[-1]-datnowall[-1]
                state = data_util.scaling(state)
                action = learner.select_action_norandom(state)
                reward = data_util.calcreward_bt(action,diff_io,bet,gain)
                moneynow+= reward
                if action != 0:
                    numTAs += 1
                    j += wait
                else:
                    j+= 1
    print '$', moneynow, ', ', numTAs/(numchances/(60*60/2)),' bets per hour'
