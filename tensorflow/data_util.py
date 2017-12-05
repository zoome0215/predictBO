from __future__ import division

import numpy as np
import os, os.path

class tradedata:
    def __init__(self):
        self.exist_data = False

    def loaddata(self,year,month,i):
        datadir = '../tickdata/twosecdata/'
        fname=datadir+'data'+str(year) + "-" + str(month).zfill(2) + "-" + str(i).zfill(4)+".out"
        if (os.path.isfile(fname)) :
            vals = []
            data=open(fname,'r')
            qq  = 0 
            for line in data:
                if np.mod(qq,2)==0 :
                    currdate,val = line.split('\t')
                    vals.extend([float(val)])
                qq += 1
            self.__data = vals
            self.exist_data=True
        else:
            self.exist_data=False

    def is_empty(self):
        return False if (self.exist_data) else True

    def size(self):
        if self.is_empty():
            return 0
        else:
            return len(self.__data)
    def get(self,i,interval):
        return np.array(self.__data[i:i+interval])

def scaling(vec):
    mu = np.mean(vec)
    std = np.std(vec)

    return ((vec-mu)/std)

def calcreward(actionnow, diff_io,bet,gain,loss):
    upparam = 1
    downparam = -1
    restparam = 0
    if actionnow == restparam:
        return -np.abs(loss)
    else:
        if actionnow == upparam:
            if diff_io > 0 :
                return -bet
            else :
                return gain
        elif actionnow == downparam:
            if diff_io < 0 :
                return gain
            else :
                return -bet
        else :
            return 0

def calcreward_bt(actionnow,diff_io,bet,gain):
    upparam = 1
    downparam = -1
    restparam = 0
    if actionnow == restparam:
        return 0
    else:
        if actionnow == upparam:
            if diff_io > 0 :
                return -bet
            else :
                return gain
        elif actionnow == downparam:
            if diff_io < 0 :
                return gain
            else :
                return -bet
        else :
            return 0
