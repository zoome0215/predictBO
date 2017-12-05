from __future__ import division
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

    meanseconds = 2
    
    endvals=[]
    for j in range(-int(meanseconds/2)-1,int(meanseconds/2)):
        endvals.append([outsnow[bi+j]])

    endvals = np.array(endvals)
    endvals = np.squeeze(endvals)
    endval = np.mean(endvals)

    updown = 1.0 - endval

    if (updown < -res):
        return up
    elif ( updown > res):
        return down
    else :
        return otherparam

#########################################3
interval = 15*60

rest_time = 60

wait = 1*60

fitSVC = 3

betinterval = 5 #min

testres=0.001/114.0

periodint = 2
payrate = 1.85

# Load AI
AIname = 'realuod_'+str(interval)+'_'+str(betinterval)+'_skl_dip_all'
AIdir  = '../AI/skl/'

AIname = AIdir+AIname
if fitSVC == 1:
    print 'svm'
    AIname = AIname + '_rc'
elif fitSVC == 2:
    print 'chosen ensemble classifier'
    AIname = AIname + '_ens'
elif fitSVC == 3:
    print 'chosen other classifier'
    AIname = AIname + '_oth'
else :
    print 'simple tree'

clf = joblib.load(AIname) 

#load extra classifier
extraclassname = 'class_u_d_dip_all_ens'
extraclassname = AIdir + extraclassname

clf_class = joblib.load(extraclassname)
interval = int(interval/periodint)
betinterval = int(betinterval*60/periodint)

datadir = '../tickdata/twosecdata/'

conseq = 0

moneynow=200
bet=20
countTAs = 0
countwins = 0
numups = 0
numdowns = 0

print 'tested on', datetime.today()

numdat=0
length = interval+betinterval+15
for year in range(2017,2018):
    for month in range(1,13) :
        for i in range(0,10000) :
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

                if qq > 60*30:
                    numdat += qq
                    vals = np.squeeze(np.array([vals]))
                    j = 0
                    while j < (vals.shape[0]-length) :
                        numdat += 1
                        innow = vals[j:j+interval]
                        norm = np.mean(innow[-1:])
                        innow = innow/float(norm)
                        outnow=vals[j+interval:j+length]/float(norm)
                        labelnow = testlabeling(innow,outnow,betinterval,testres)
                        prednow = clf.predict(innow.reshape(1,-1))
                        #probnow = clf.predict_proba(innow.reshape(1,-1))
                        #probnow = clf.decision_function(innow.reshape(1,-1))
                        #betnow=int(clf_class.predict(probnow))
                        #betprobnow=clf_class.predict_proba(probnow)
                        #betprobnow = betprobnow[0]
                        betprobnow = [1,1]
                        betnow = 1

                        if (prednow == upparam) & (betnow==1) :
                            if (labelnow == 1)  :
                                print betprobnow,'up'
                                moneynow += (payrate*bet - bet)
                                countwins += 1
                            else :
                                print betprobnow,'up, fail'
                                moneynow -= bet
                            countTAs += 1
                            numups += 1
                            j += wait
                        elif (prednow == downparam) & (betnow==1):
                            if (labelnow == 2) :
                                print betprobnow,'down'
                                moneynow += (payrate*bet - bet)
                                countwins += 1
                            else :
                                print betprobnow,'down, fail'
                                moneynow -= bet
                            countTAs += 1
                            numdowns += 1
                            j += wait
                        else :
                            j += 1

                        if moneynow < 0 :
                            print 'Fail! The money is negative!'
                            sys.exit(0)

                    if countTAs == 0:
                        pwin = 0
                    else :
                        pwin = 100*float(float(countwins)/float(countTAs))
                    if numdat != 0:
                        print i,pwin, countTAs, 60*60*countTAs/numdat,numdat/(60*60),'hours, $',moneynow,':'
                        #print np.mean(failprob),np.std(failprob),np.mean(succprob),np.std(succprob)
                    
            else :
                break
        fname=datadir+'data'+str(year) + "-" + str(month).zfill(2) + "-" + str(0).zfill(4)+".out"
        if (os.path.isfile(fname)) :
            print year, month
            print 2*numdat/(60*60*24),'days worth of chances!'
            print countTAs,'transactions are made'
            print 'had', countwins,'wins.'
            print 'winrate is', pwin,'%'
            print '$',moneynow
            print numups,'up predictions and', numdowns,'down predictions'

