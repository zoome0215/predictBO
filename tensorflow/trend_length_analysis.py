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

payrate=2
initmoney = 200
bet = 20

test_year = 2017

countthresh = 4

possibleactions = (downparam,restparam,upparam)
weightactions = np.array([0.2,0.6,0.2])

if betinterval < 10 :
    payrate = 1.85

print 'tested on', datetime.today()
print 'interval of ' , interval ,' min'
print 'loading AI...'

#Converting the intervals to array length
interval = int(interval*60/periodint)
betinterval = int(betinterval*60/periodint)

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

uplengths= []
downlengths = []
restlengths = []

actionprev  = restparam
conseccount = 1

for month in range(1,2):
    for i in range(0,1000):
        datanow.loaddata(test_year,month,i)
        if (not datanow.exist_data) :
            break
        jlim = datanow.size()-(interval+betinterval)
        if jlim > (interval) :
            j=0
            numchances += jlim

            while j < jlim :

                datnowall = datanow.get(j,interval+betinterval)
                state = datnowall[:interval]
                diff_io=state[-1]-datnowall[-1]
                state = data_util.scaling(state)
                action = learner.select_action_norandom(state)

                if action == actionprev :
                    conseccount += 1
                else :
                    if actionprev == upparam:
                        uplengths.append([conseccount])
                    elif actionprev == downparam:
                        downlengths.append([conseccount])
                    elif actionprev == restparam:
                        restlengths.append([conseccount])
                    
                    conseccount=1
                    actionprev = action

                j+= 1


            if checkQ:
                count += 1
                if count > countthresh:
                    Qchecked = True

            if Qchecked :
                print 'after', count, 'files:'
                print 'average up',np.mean(uplengths)
                print 'average down',np.mean(downlengths)
                print 'average rest',np.mean(restlengths)
                sys.exit(0)

print 'after', count, 'files:'
print 'average up',np.mean(uplengths)
print 'average down',np.mean(downlengths)
print 'average rest',np.mean(restlengths)
sys.exit(0)
