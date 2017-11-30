#!/usr/bin/python
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

from sklearn.linear_model import SGDClassifier
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.externals import joblib

import matplotlib.pyplot as plt

upparam=5
downparam=6
otherparam=0

def loaddata(interval,predinterval,rest_time):
    length = interval+predinterval
    traindata = []
    teacherdata = []
    datadir = './tickdata/twosecdata/'
    for year in range(2015,2016):
        for month in range(1,12) :
            for i in range(0,10000) :
                fname=datadir+'data'+str(year) + "-" + str(month).zfill(2) + "-" + str(i).zfill(4)+".out"
                if (os.path.isfile(fname)) :
                    data=open(fname,'r')
                    vals = []
                    qq  = 0 
                    for line in data:
                        if np.mod(qq,2)==0 :
                            currdate,val = line.split('\t')
                            vals.extend([float(val)])
                        qq += 1

                    if qq > 2:
                        vals = np.squeeze(np.array([vals]))
                        j = 0
                        while j < len(vals)-length :
                            #traindata.append(vals[j:j+interval])
                            #teacherdata.append(vals[j+interval:j+length])
                            datnow = vals[j:j+interval]
                            norm = np.mean(datnow[-1:])
                            traindata.append(datnow/float(norm))
                            teacherdata.append(vals[j+interval:j+length]/float(norm))
                            j += np.random.randint(rest_time)+1
                    data.close()
                else :
                    break

    traindata = np.array(traindata)
    traindata = np.squeeze(traindata)
    teacherdata = np.array(teacherdata)
    teacherdata = np.squeeze(np.array(teacherdata))
    print float(np.shape(traindata)[0])*2.0/(60.0*60.0*24),'days of data are used for training'
    return traindata,teacherdata

def testloaddata(interval,predinterval,rest_time):
    length = interval+predinterval
    traindata = []
    teacherdata = []
    datadir = './tickdata/twosecdata/'
    for year in range(2015,2016):
        for month in range(12,13) :
            for i in range(0,10000) :
                fname=datadir+'data'+str(year) + "-" + str(month).zfill(2) + "-" + str(i).zfill(4)+".out"
                if (os.path.isfile(fname)) :
                    data=open(fname,'r')
                    vals = []
                    qq  = 0 
                    for line in data:
                        if np.mod(qq,2)==0 :
                            currdate,val = line.split('\t')
                            vals.extend([float(val)])
                        qq += 1

                    if qq > 2:
                        vals = np.squeeze(np.array([vals]))
                        j = 0
                        while j < len(vals)-length :
                            #traindata.append(vals[j:j+interval])
                            #teacherdata.append(vals[j+interval:j+length])
                            datnow = vals[j:j+interval]
                            norm = np.mean(datnow[-1:])
                            traindata.append(datnow/float(norm))
                            teacherdata.append(vals[j+interval:j+length]/float(norm))
                            j += np.random.randint(rest_time)+1
                    data.close()
                else :
                    break

    traindata = np.array(traindata)
    traindata = np.squeeze(traindata)
    teacherdata = np.array(teacherdata)
    teacherdata = np.squeeze(np.array(teacherdata))
    print float(np.shape(traindata)[0])*2.0/(60.0*60.0*24),'days of data are used for training'
    return traindata,teacherdata

def labeling(train,outs,oi,bi,offsetint,res,perin):
    
    up=1
    down=2
    
    other=0

    numtrain = train.shape[0]

    labels=[]
    unlabeled=0

    res1 = res/2.0
    res2 = res/4.0
    for i in range(0,numtrain):
        feednow=train[i,:]
        outsnow=outs[i,:]

        outsnow = np.array(outsnow)
        meanval = np.mean(outsnow[bi-2:bi+1])

        meanval = 1.0 - meanval
        outsnow = outsnow[offsetint:oi+offsetint] 
        outsnow = 1.0 - outsnow

        if (  (outsnow[outsnow < -res].size/outsnow.size) > perin) & (meanval < (-res) ):
            labels.append([upparam])
        elif ( (outsnow[outsnow < -res1].size/outsnow.size) > perin) & (meanval <-res1):
            labels.append([3])
        elif ( (outsnow[outsnow < -res2].size/outsnow.size) > perin) & (meanval <-res2):
            labels.append([up])
        elif ( (outsnow[outsnow > res].size/outsnow.size) > perin) & (meanval > res):
            labels.append([downparam])
        elif ( (outsnow[outsnow > res1].size/outsnow.size) > perin ) & (meanval > res1)   :
            labels.append([4])
        elif ( (outsnow[outsnow > res2].size/outsnow.size) > perin ) & (meanval > res2)   :
            labels.append([down])
        else :
            unlabeled += 1
            labels.append([otherparam])
    print 'percent data used:',100.0*unlabeled/len(labels),'%'
    labels = np.array(labels)
    return labels

def testlabeling(train,outs,bi,res):
    
    up=1
    down=2
    
    upstrong=3
    downstrong=4

    upstronger=5
    downstronger=6

    other=0

    conseqthresh=20
    conseqthresh2=40

    numtrain = train.shape[0]

    labels=[]
    conseqs = []
    beginds = []

    labeled=0
    conseq=0
    lastlabel=0
    for i in range(0,numtrain):
        feednow=train[i,:]
        outsnow=outs[i,:]

        meanseconds=2
        endvals=[]
        for j in range(-int(meanseconds/2)-1,int(meanseconds/2)):
            endvals.append([outsnow[bi+j]])

        endvals = np.array(endvals)
        endvals = np.squeeze(endvals)
        endval = np.mean(endvals)

        updown = 1.0 - endval

        if (updown < -res):
            if lastlabel == up :
                if int(conseqs[-1][0]) == 0:
                    beginds.append([i])
                conseqs[-1] = [conseqs[-1][0]+1]
            else :
                conseqs.append([0])
                beginds.append([i])
            labeled += 1.0
            labels.append([upparam])
        elif ( updown > res):
            if lastlabel == down :
                if int(conseqs[-1][0]) == 0 :
                    beginds.append([i])
                conseqs[-1] = [conseqs[-1][0]+1]
            else :
                conseqs.append([0])
                beginds.append([i])
            labeled += 1.0
            labels.append([downparam])
        else :
            labels.append([otherparam])

        lastlabel=labels[-1]
        lastlabel=lastlabel[0]

    labels = np.array(labels)
    return labels

def AItester(testdata,testlabels,testpred,initmoney,initbet,payrate,verbose):
    moneynow = initmoney
    bet = initbet
    countTAs = 0
    countwins = 0

    conseq = 0
    predbefore = 0

    numups = 0
    numdowns = 0

    wait = int(5*60/2.0)
    j = 0
    conseq = 0
    cumstd = []
    countcumstd = 0
    while j < np.shape(testdata)[0]:
        datnow = testdata[j,:]
        if np.std(datnow) < 5.000 : #Volatility
            prednow = testpred[j]
            if (prednow == upparam) :
                if (j+ conseq) < np.shape(testdata)[0] :
                    labelnow = int(testlabels[j+conseq])
                else :
                    break
                if (labelnow == upparam) :
                    cumstd.append(np.std(datnow))
                    moneynow += (payrate*bet - bet)
                    countwins += 1
                else :
                    moneynow -= bet
                countTAs += 1
                numups += 1
                j += wait
            elif (prednow == downparam):
                if (j+ conseq) < np.shape(testdata)[0] :
                    labelnow = int(testlabels[j+conseq])
                else :
                    break
                if (labelnow == downparam) :
                    cumstd.append(np.std(datnow))
                    moneynow += (payrate*bet - bet)
                    countwins += 1
                else :
                    moneynow -= bet
                numdowns += 1
                countTAs += 1
                j += wait
            else :
                j += 1
        else :
            j+=1

    if countTAs == 0:
        pwin = 0
    else :
        pwin = 100*float(float(countwins)/float(countTAs))

    perhour  =  0.5*60.0*60.0*float(countTAs)/float(np.shape(testdata)[0])

    if verbose == 1:
        print 'out of',np.shape(testdata)[0],'chances, which is',float(np.shape(testdata)[0])*2.0/(60.0*60.0),'hours'
        print countTAs,'transactions are made'
        print 'had',countwins,'wins.'
        print 'winrate is',pwin,'%'
        print '$',initmoney,'became $',moneynow,'at the end of',float(np.shape(testdata)[0])*2.0/(60.0*60.0),'hours'
        print perhour, 'per hour, consists of ',numups,'up predictions and',numdowns,'down predictions'
    return countTAs, countwins, pwin, moneynow, perhour


######################################

interval = 15*60

res = 0.02 #15, 0.05

perin = 0.99999

numDTC = 20

rest_time = int(5*60/2)

noAI = 1

fitSVC = 3

force = 1

testmode = 0

save = 1

offsetint = int(60.0*3.5/2.0)

betinterval = 5 #min
dipinterval = 3 #min

testres1=0.005/114.0
testres2=0.003/114.0
testres3=0.001/114.0

#####################################

#####################################

periodint = 2

#data
begind = 0
endind = 6
numhours = 40

verbose = 1

testmode = 0
testing = 1
showplot = 0

#testres=res/114
res=res/114.0 

payrate=2

if betinterval < 10 :
    payrate = 1.85

initmoney = 200
bet = 20

# old was outname = 'upordown_'+str(betinterval) # betinterval = 5
outname = 'realuod_'+str(interval)+'_'+str(betinterval)+'_skl_dip_all'
interval=interval/60.0
print 'interval of ' , interval ,' min'
print 'with ', numDTC , ' classifiers'

interval = int(interval*60/periodint)

betinterval = int(betinterval*60/periodint)
dipinterval = int(dipinterval*60/periodint)

print 'tested on', datetime.today()
print 'making AI...'
print 'Resolution is', res

if testmode != 1:
    #load train data
    print 'loading data'
    traindata,outs = loaddata(interval,dipinterval+offsetint+15,rest_time)
    print 'data loading done!'

    labels = labeling(traindata,outs,dipinterval,betinterval,offsetint,res,perin)

#load test data
print 'loading test data'
testdata,testouts = testloaddata(interval,betinterval+15,1)
print 'done loading test data'

testlabels = testlabeling(testdata,testouts,betinterval,testres1)

'''
num0s = np.mean([sum(labels==upparam),sum(labels==downparam)])
num0s = 5*num0s
num0s = int(num0s)
if labels[ labels == 0].size > num0s:
    print 'randomised!',labels[ labels == 0].size, 'is shrunk to',num0s
    randarr = np.random.choice(labels[labels==0].size,num0s)
    traindata = np.r_[traindata[np.squeeze(labels==0),:][randarr,:],traindata[np.squeeze(labels==upparam),:],traindata[np.squeeze(labels==downparam),:]]
    labels = np.r_[labels[labels==0][randarr],labels[labels==upparam],labels[labels==downparam]]
'''


#reshping
if testmode != 1:
    traindata = np.reshape(traindata,[-1,interval])
testdata = np.reshape(testdata,[-1,interval])

#making AI
outdir  = './AI/skl/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
outname = outdir+outname
if fitSVC == 1:
    outname = outname + '_rc'
elif fitSVC == 2:
    print 'chosen ensemble classifier'
    outname = outname + '_ens'
elif fitSVC == 3:
    print 'chosen other classifier'
    outname = outname + '_oth'
else :
    print 'simple tree'

if (os.path.isfile(outname)) & (force != 1) :
    clf = joblib.load(outname) 

    print 'now predicting...'
    predicted = clf.predict(testdata)
    print 'done!'

    print 'predict test 1'
    testlabels = testlabeling(testdata,testouts,betinterval,testres1)
    countTAs,countwins,pwin,moneynow,perhour = AItester(testdata,testlabels,predicted,initmoney,bet,payrate,1)

    print 'predict test 3'
    testlabels = testlabeling(testdata,testouts,betinterval,testres3)
    countTAs,countwins,pwin1,moneynow,perhour = AItester(testdata,testlabels,predicted,initmoney,bet,payrate,1)

else :
    pwin=0

pwinmax = pwin
if testmode != 1:
    print 'current max prob:',pwinmax
    for i in range(0,1000):
        print i
        if noAI == 1:
            print 'now fitting...'
            if fitSVC == 1:
                clf = linear_model.RidgeClassifier(alpha=1e-15,copy_X=False,tol=1e-16)
                #clf = svm.LinearSVC(decisi{n_function_shape='ovr',verbose=1)
                clf.fit(traindata,labels.ravel())
            elif fitSVC == 2:
                clf = ensemble.BaggingClassifier(ExtraTreeClassifier(),n_estimators=numDTC)
                clf.fit(traindata,labels.ravel())
            elif fitSVC == 3:
                clf = neighbors.KNeighborsClassifier(n_neighbors=20)
                clf.fit(traindata,labels.ravel())
            else :
                clf = DecisionTreeClassifier(max_depth=25,min_samples_leaf=10)
                clf.fit(traindata,labels)
            print 'done!'
        else :
            clf = joblib.load(outname) 

        if testing == 0:
            sys.exit(0)

        if noAI == 1:

            print 'now predicting...'
            predicted = clf.predict(testdata)
            print 'done!'

            testlabels = testlabeling(testdata,testouts,betinterval,testres1)
            countTAs,countwins,pwin,moneynow,perhour = AItester(testdata,testlabels,predicted,initmoney,bet,payrate,1)
        if force == 1:
            pwin += 100

        if (pwin > pwinmax) | (noAI != 1):
            pwinmax0 = pwinmax
            pwinmax = pwin
            testlabels = testlabeling(testdata,testouts,betinterval,testres1)
            countTAs,countwins,pwin1,moneynow,perhour = AItester(testdata,testlabels,predicted,initmoney,bet,payrate,1)
            print '\n'
            testlabels = testlabeling(testdata,testouts,betinterval,testres2)
            countTAs,countwins,pwin2,moneynow,perhour = AItester(testdata,testlabels,predicted,initmoney,bet,payrate,1)
            print '\n'
            testlabels = testlabeling(testdata,testouts,betinterval,testres3)
            countTAs,countwins,pwin3,moneynow,perhour = AItester(testdata,testlabels,predicted,initmoney,bet,payrate,1)
            if (noAI == 1) & (pwin > pwinmax0) & (save != 0):
                print 'max!'
                joblib.dump(clf, outname) 
                print 'saved!'

        if (noAI != 1) | (fitSVC != 2) | (force == 1):
            break
