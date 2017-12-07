from __future__ import division
import sys
import os
import numpy as np
import datetime
import pytz

year=2017
datadir="../../data/tickdata/twosecdata/"

for year in range(1999,2020):
    for i in range(1,13):
        truncatedfname=datadir + "data"+str(year)+"-"+str(i).zfill(2)+"-0000.out"
        if not os.path.isfile(truncatedfname) :
            fname="../tickdata/truncateddata/data"+str(year)+"-"+str(i).zfill(2)+".txt"
            if os.path.isfile(fname) :
                print year,i
                data = open(fname)
                dates = [] 
                vals = []
                for line in data :
                    currdate,midval = line.split(',' )
                    midval = float(midval)
                    currdate = datetime.datetime.strptime(currdate,'%Y-%m-%d %H:%M:%S')
                    dates.append(currdate)
                    vals.append(midval)
                j = 0
                outvals=[]
                outdates=[]
                for k in range(0,len(vals)) :
                    outname="../tickdata/twosecdata/data"+str(year)+"-"+str(i).zfill(2)+"-"+str(j).zfill(4)+".out"
                    outvals.append(vals[k])
                    outdates.append(dates[k])
                    if (k + 1) < len(vals) :
                        if (dates[k+1] - dates[k]).seconds > 60 :
                            fID = open(outname,'w')
                            for ll in range(0,len(outvals)):
                                str1=outdates[ll].strftime("%Y-%m-%d %H:%M:%S")
                                fID.write("%s\t%f\n"%(str1,outvals[ll]))
                            fID.close()
                            outvals=[]
                            outdates=[]
                            j += 1
                        elif (dates[k+1] - dates[k]).seconds > 1:
                            currdate = dates[k]
                            dv = (vals[k+1] - vals[k])/ (dates[k+1] - dates[k]).seconds 
                            ss = 1
                            while (dates[k+1]-currdate).seconds > 1  :
                                currdate = currdate  +  datetime.timedelta(0,1)
                                currval= vals[k] + ss * dv
                                outvals.append(currval)
                                outdates.append(currdate)
                                ss += 1
                fID = open(outname,'w')
                for ll in range(0,len(outvals)):
                    str1=outdates[ll].strftime("%Y-%m-%d %H:%M:%S")
                    fID.write("%s\t%f\n"%(str1,outvals[ll]))
                fID.close()
                outvals=[]
                outdates=[]
                j += 1
        else :
            print "the file already exists!"

