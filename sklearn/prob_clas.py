#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import numpy as np
import os, os.path
import sys
import time
import pickle

from datetime import datetime

import sklearn.svm as svm

from sklearn import linear_model
from sklearn import ensemble

from sklearn import neighbors

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

from sklearn.externals import joblib

import matplotlib.pyplot as plt

upparam=5
downparam=6
otherparam=0

def testlabeling(feednow,outsnow,bi,res):
    
    up=1
    down=2

    meanseconds=2

    endvals=[]
    for j in range(-int(meanseconds/2)-1,int(meanseconds/2)):
        endvals.append([outsnow[bi+j]])

    endvals = np.array(endvals)
    endvals = np.squeeze(endvals)
    endval = np.mean(endvals)

    updown = 1.0 - endval

    if (updown < (-res)):
        return up
    elif ( updown > res): 
        return down
    else :
        return otherparam

#########################################3
interval = 15*60

rest_time = 60

wait = 60

betinterval = 5 #min

outname = 'class_u_d_dip_all_ens'

AIsuffix = '_oth'

tp = 0.000

testres=0.001/114.0

numDTC = 10


periodint = 2
payrate = 1.85

# Load AI
AIdir  = '../AI/skl/'
AIname = 'realuod_'+str(interval)+'_'+str(betinterval)+'_skl_dip_all'
AIname = AIname + AIsuffix
AIname = AIdir+AIname

clf = joblib.load(AIname) 

#Create AI
outname = AIdir + outname

#clf_out = DecisionTreeClassifier()
#clf_out = linear_model.RidgeClassifier(alpha=1e-15,copy_X=False,tol=1e-16)
#clf_out = svm.SVC(C=100,probability=True,tol=1e-5,verbose=1)
#clf_out = ensemble.BaggingClassifier(ExtraTreeClassifier(),n_estimators=numDTC)
clf_out = neighbors.KNeighborsClassifier()
#clf_out = ensemble.RandomForestClassifier(n_estimators=numDTC)

#real thing
interval = int(interval/periodint)
betinterval = int(betinterval*60/periodint)
datadir = '../tickdata/twosecdata/'

moneynow=200
bet=20
countTAs = 0
countwins = 0
numups = 0
numdowns = 0

print('tested on', datetime.today())
print(AIname)
print(outname)

#initialization
numdat=0
length = interval+betinterval+15
probs = []
succs = []

#main loop
for year in range(2016,2017):
    for month in range(1,13) :
        for i in range(0,1000) :
            fname=datadir+'data'+str(year) + "-" + str(month).zfill(2) + "-" + str(i).zfill(4)+".out"
            if (os.path.isfile(fname)) :
                data=open(fname,'r')
                vals = []
                qq  = 0 
                for line in data:
                    if np.mod(qq,periodint)==0:
                        currdate,val = line.split('\t')
                        vals.extend([float(val)])
                    qq += 1
                data.close()
                if qq > (60*30) :
                    print ('optimizing with', qq, 'sequences')
                    numdat += qq
                    vals = np.squeeze(np.array([vals]))
                    j = 0
                    jmax = vals.shape[0]-length
                    percent_complete = 0
                    while j < jmax :
                        printstr = ' '
                        if (j/qq) >= percent_complete :
                            print (' '*(len(printstr)+1),end='\r')
                            sys.stdout.flush()
                            printstr ='{0:3.1f} percent completed!'.format(100.0*j/jmax) 
                            print (printstr,end='\r')
                            sys.stdout.flush()
                            percent_complete += ((j/qq)*1000)//1000+0.001
                        innow = vals[j:j+interval]
                        norm = np.mean(innow[-1:])
                        innow = innow/float(norm)
                        outnow=vals[j+interval:j+length]/float(norm)
                        labelnow = testlabeling(innow,outnow,betinterval,testres)
                        prednow = clf.predict(innow.reshape(1,-1))
                        probnow = clf.predict_proba(innow.reshape(1,-1))
                        #probnow = clf.decision_function(innow.reshape(1,-1))
                        probnow = probnow[0]

                        if (prednow == upparam) :
                            probs.append(probnow)
                            if (labelnow == 1):
                                succs.append([int(1)])
                                moneynow += (payrate*bet - bet)
                                countwins += 1
                            else :
                                succs.append([0])
                                moneynow -= bet
                            countTAs += 1
                            numups += 1
                            j += wait
                        elif (prednow == downparam) :
                            probs.append(probnow)
                            if (labelnow == 2) :
                                succs.append([int(1)])
                                moneynow += (payrate*bet - bet)
                                countwins += 1
                            else :
                                succs.append([int(0)])
                                moneynow -= bet
                            countTAs += 1
                            numdowns += 1
                            j += wait
                        else :
                            j += 1

                    if countTAs == 0:
                        pwin = 0
                    else :
                        pwin = 100*float(float(countwins)/float(countTAs))
                    if numdat != 0:
                        print (i,pwin, countTAs, 60*60*countTAs/numdat,numdat/(60*60),'hours, $',moneynow,':',tp)
                        #print np.mean(failprob),np.std(failprob),np.mean(succprob),np.std(succprob)
                    
            else :
                break
        fname=datadir+'data'+str(year) + "-" + str(month).zfill(2) + "-" + str(0).zfill(4)+".out"
        if (os.path.isfile(fname)) :
            print (year, month)
            print (2*numdat/(60*60*24),'days worth of chances!')
            print (countTAs,'transactions are made')
            print ('had', countwins,'wins.')
            print ('winrate is', pwin,'%')
            print ('$',moneynow)
            print (numups,'up predictions and', numdowns,'down predictions')

probs = np.array(probs)
succs = np.array(succs)


probs = np.reshape(probs,[-1,probs.shape[1]])
clf_out.fit(probs,succs)
joblib.dump(clf_out, outname)

print ('extra classifier outputted!!!!')
print (probs.shape,succs.shape)
