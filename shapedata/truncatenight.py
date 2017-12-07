import os
import numpy as np
import datetime
import pytz

datadir="../data/tickdata/twosecdata/"
for year in range(2010,2020) :
    for i in range(1,13):
        truncatedfname=datadir + "data"+str(year)+"-"+str(i).zfill(2)+"-0000.out"
        if not os.path.isfile(truncatedfname) :
            print year,i 
            fname="../data/tickdata/rawdata/DAT_NT_USDJPY_T_LAST_"+str(year)+str(i).zfill(2)+".csv"
            outname="../data/tickdata/truncateddata/data"+str(year)+"-"+str(i).zfill(2)+".txt"
            if os.path.isfile(fname) :
                print 'outputting'
                outdata = open(outname, "w")
                data = open(fname)
                for line in data:
                    currdate,midval,nullval = line.split(';' )
                    midval = float(midval)
                    nullval = int (nullval)
                    currdate = datetime.datetime.strptime(currdate,'%Y%m%d %H%M%S')
                    currdate = pytz.timezone('US/Eastern').localize(currdate)
                    currdate = currdate.astimezone(pytz.timezone('Asia/Tokyo'))
                    if (currdate.weekday() < 5):
                        if (currdate.time() > datetime.time(8,00) ) & ( currdate.time() <= datetime.time(23,59)):
                            outdata.write('%s,%f\n'%(currdate.replace(tzinfo=None),midval))
                        elif (currdate.time() >= datetime.time(0,00) ) & ( currdate.time() < datetime.time(4,00)):
                            outdata.write('%s,%f\n'%(currdate.replace(tzinfo=None),midval))
                print(currdate)
                outdata.close()
        else :
            print "the file already exists!"


