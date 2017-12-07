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
cont_learn = True

interval = 15 # min 
betinterval = 5 # min
periodint = 2

outname = 'realuod_'+str(interval)+'_'+str(betinterval)+'_tf'

target_rate = 4 # per hour
payrate=2
initmoney = 200
bet = 20

lr0 =   7.31165891657e-06
greed0= 0.63707365637

lrthresh = 1e-8
epsthresh = 0.1

train_year = 2016
Nepochs = 10000

wait = 5 #min

possibleactions = (downparam,restparam,upparam)
#weightactions = np.array([0.2,0.6,0.2])
weightactions = np.array([0.4,0.2,0.4])

if betinterval < 10 :
    payrate = 1.85

gain = bet*(payrate-1)
#loss = (payrate-1)*(periodint*bet*target_rate/(60.0*60.0))/2.0
loss=0.0

print 'tested on', datetime.today()
print 'interval of ' , interval ,' min'
print 'loss is', loss

#Converting the intervals to array length
interval = int(interval*60/periodint)
betinterval = int(betinterval*60/periodint)
wait = int(wait*60/2)

#making AI
outdir  = '../AI/tf/'
outname = outdir+outname

learner = learning_util.learn(outname,possibleactions,weightactions,interval)

#load data
datanow = data_util.tradedata()

if cont_learn:
    print 'AI loaded!'
    learner.loadmodel();
else:
    print 'Making a new AI!'

numdata=0
numtrials = 0
numlearns = 0

moneynow=initmoney
actioncount = 0
lastaction = 1

for e in range(Nepochs):
    if e%100 ==0:
        print 'epoch ', e
    for month in range(1,13):
        for i in range(0,1000):
            datanow.loaddata(train_year,month,i)
            if (not datanow.exist_data) :
                break
            jlim = datanow.size()-(interval+max(betinterval,wait))
            if jlim > (wait+interval):
                currepsilon = greed0*np.exp(-numlearns/100000.0)
                currlr = lr0*np.exp(-numlearns/10000.0)

                if currepsilon < epsthresh :
                    currepsilon = epsthresh
                if currlr < lrthresh :
                    currlr = lrthresh

                learner.set_epsilon(currepsilon)
                learner.set_lr(currlr)
                numdata+=jlim

                j=0
                while j < jlim :
                    datnowall = datanow.get(j,interval+max(betinterval,wait))
                    state = datnowall[:interval]
                    diff_io = state[-1]-datnowall[interval+betinterval-1]
                    state = data_util.scaling(state)
                    action = learner.select_action(state, learner.exploration)
                    reward = data_util.calcreward(action,diff_io,bet,gain,loss)

                    if action == 0:
                        statenext = datnowall[1:interval+1]
                    else :
                        statenext = datnowall[wait:(interval+wait)]

                    if (lastaction == 0) and (action == 0) :
                        actioncounter += 1
                    else :
                        actioncounter = 0

                    if actioncounter > (10*60/2) :
                        actioncounter = 0
                        reward =  -((payrate-1)*(periodint*bet*target_rate)/2.0)


                    moneynow+= reward

                    if moneynow < 0:
                        terminal=True
                        moneynow = initmoney
                    else :
                        terminal=False

                    statenext = data_util.scaling(statenext)
                    learner.storeexperience(state,action,reward,statenext,terminal)

                    if (numtrials%(60*60*1/periodint)) == 0:
                        learner.experience_replay()
                        numlearns+= 1


                    if action == 0:
                        j+= 1
                    else :
                        j+=wait

                    numtrials += 1

                    lastaction = action

                print train_year,month, i, 'learned', numlearns,'times with epsilon =',currepsilon, \
                        'and lr =', currlr
        learner.savemodel()
