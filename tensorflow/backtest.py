#!/usr/bin/python
from __future__ import division

import math
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

#plotting
import matplotlib.pyplot as plt

upparam = 1
downparam = -1
restparam = 0

#############################
checkQ = True

interval = 15 # min 
betinterval = 5 # min
periodint = 2

outname = 'realuod_'+str(interval)+'_'+str(betinterval)+'_tf'

target_rate = 4 # per hour
payrate=2
initmoney = 200
bet = 20

test_year = 2017

wait = 10 #min

countthresh = 10
Qthresh = 1.5

if countthresh > 5:
    checkQ = False

possibleactions = (downparam,restparam,upparam)
weightactions = np.array([0.2,0.6,0.2])

if betinterval < 10 :
    payrate = 1.85

gain = bet*(payrate-1)

print 'tested on', datetime.today()
print 'interval of ' , interval ,' min'
print 'Qthresh =',Qthresh

print 'loading AI...'

#Converting the intervals to array length
interval = int(interval*60/periodint)
betinterval = int(betinterval*60/periodint)
wait = int(wait*60/periodint)

#making AI
outdir  = '../AI/tf/'
outname = outdir+outname


learner = learning_util.learn(outname,0.001,possibleactions,weightactions,interval)
learner.loadmodel()

#load data
datanow = data_util.tradedata()

moneynow=initmoney
numchances = 0 
numTAs = 0

Qchecked = False
count = 0
Qvals=[]
prof=0
for month in range(1,5):
    for i in range(0,1000):
        datanow.loaddata(test_year,month,i)
        if (not datanow.exist_data) :
            break
        jlim = datanow.size()-(interval+betinterval)
        if jlim > (wait+interval) :
            j=0
            numchances += jlim

            while j < jlim :

                datnowall = datanow.get(j,interval+betinterval)
                state = datnowall[:interval]
                diff_io=state[-1]-datnowall[-1]
                state = data_util.scaling(state)
                action = learner.select_action_norandom(state)

                Qnow = max(learner.Q_values(state))
                if checkQ :
                    Qvals.append(learner.Q_values(state))

                reward = math.floor(data_util.calcreward(action,diff_io,bet,gain))
                if (action != 0):
                    if (Qnow > Qthresh):
                        moneynow += reward
                        if reward >0 :
                            print 'success', Qnow
                        else :
                            print 'fail', Qnow
                    else :
                        moneynow += 0
                else :
                    moneynow+= reward

                if (action != 0) and (Qnow > Qthresh):
                    numTAs += 1
                    j += wait
                else:
                    j+= 1

                if moneynow > 1000:
                    bet=math.floor(0.10*moneynow)
                    if bet > 500:
                        bet = 500
                else :
                    bet=20

                if moneynow < 0:
                    print 'money became negative'
                    print test_year, month, i, '$', moneynow, ', ', numTAs,'transactions, which is', \
                            numTAs/(numchances/(60*60/2)), \
                        ' bets per hour', 'over', (numchances/(60*60/periodint)), 'hours'
                    print 'Stopped at step', j
                    sys.exit(0)

            if checkQ:
                count += 1
                if count > countthresh:
                    Qchecked = True

            print test_year, month, i, '$', moneynow, \
                    ', ', numTAs,'transactions, which is', numTAs/(numchances/(60*60/2)), \
                ' bets per hour', 'over', (numchances/(60*60/periodint)), 'hours'
            if Qchecked :
                Qvals = np.array(Qvals)
                Qvals = np.squeeze(Qvals)
                plt.plot(range(0,Qvals.shape[0]),Qvals[:,0],label='down')
                plt.plot(range(0,Qvals.shape[0]),Qvals[:,1],label='stay')
                plt.plot(range(0,Qvals.shape[0]),Qvals[:,2],label='up')
                plt.legend()
                plt.show()
                sys.exit(0)
