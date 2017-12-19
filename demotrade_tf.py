#!/usr/bin/python
from __future__ import division

import numpy as np
import os, os.path
import sys

from datetime import datetime
import time

from selenium import webdriver

import tensorflow as tf
sys.path.insert(0, './tensorflow/')
import data_util
import learning_util

upparam = 1
downparam = -1
restparam = 0

def terminateall(driver):
    driver.quit()
    sys.exit(0)
    return 0

def checktradetime():
    timenow = datetime.now()
    if ((timenow.hour >= 5) and (timenow.hour < 10) ):
        print 'It\'s not 9 yet!!!'
        print 'See you tomorrow!'
        return True
    return False

def initialization(interval,periodint,currvaltext):
    print 'now initializing for interval =',interval,'minutes...'
    time_list = []
    val_list = []
    for j in range(0,int(interval*60/periodint)):
        currval = float(currvaltext.text)
        time_list.extend([datetime.now()])
        val_list.extend([currval])
        if (j*periodint % 60) == 0:
            print j*periodint,'seconds elapsed'
        time.sleep(periodint)
    return time_list,np.squeeze(np.array([val_list]))
    print 'done initializing'

def makeinput(interval,periodint,val_list):
    begind = int(interval*60/periodint)
    datin=np.squeeze(val_list[-begind:]/float(np.mean(val_list[-2:])))
    return datin

def output_data(time_list,val_list,num):
    outdir = './realdata/'
    outname= outdir+'data-'+str(int(num)).zfill(4)+'.out'
    val_list = np.squeeze(val_list)

    fID = open(outname,'w')
    for j in range(0,len(val_list)):
        str1=time_list[j].strftime("%Y-%m-%d %H:%M:%S")
        fID.write("%s\t%f\n"%(str1,val_list[j]))

    fID.close()
    return 0

def makebet(pred,bet_field,up_button,down_button,bet_button,bet):
    bet_field.clear()
    bet_field.send_keys(bet)
    if int(pred) == downparam:
        down_button.click()
        bet_button.click()
    elif int(pred) == upparam:
        up_button.click()
        bet_button.click()
    return 0

################################################################

currpair = 'USD/JPY'
interval = 15 #mins
betinterval = 5 #mins
periodint = 2 #seconds
maxmin = 60*50
waittime = 120 # in seocnds

betamount = 20

fitSVC = 1

AIname = 'realuod_'+str(interval)+'_'+str(betinterval)+'_tf'

################################################################
if(checktradetime()):
    sys.exit(0)

clickbuttontest=0
  
driver = webdriver.Chrome('./chromedriver')
driver.get('https://en.demotrade.highlow.net/Trading');
driver.maximize_window()
time.sleep(2)
driver.find_element_by_xpath('//*[@id="header"]/div/div/div/div/div/span/span/a[1]').click() # Jump to demo account
time.sleep(6) 
#driver.find_element_by_xpath('//*[@id="account-balance"]/div[2]/div/div[1]/a').click() # delete the annoying message of cash back
driver.find_element_by_xpath('/html/body/div[6]').click() # delete the annoying message of cash back
time.sleep(6) 
driver.find_element_by_id('ChangingStrikeOOD').click() #change to Turbo
time.sleep(6) 
driver.find_element_by_xpath('//*[@id="assetsCategoryFilterZoneRegion"]/div/div[14]').click() #change interval (it should be 5m) 
driver.find_element_by_xpath('//*[@id="2225"]/div/div[2]/div[1]/div[2]/div').click() #change currency (it should be USD/JPY)

assetname=driver.find_element_by_xpath('//*[@id="asset"]')
betintervalname=driver.find_element_by_xpath('//*[@id="2225"]/div/div[1]/div[2]/span')

print betintervalname.text
print assetname.text

if (assetname.text != currpair) or (betintervalname.text != str(betinterval)+'m'):
    print 'currency and interval don\'t match!'
    terminateall(driver)

driver.execute_script("window.scrollTo(0, 280)")

bet_button = driver.find_element_by_xpath('//*[@id="invest_now_button"]')
bet_field = driver.find_element_by_xpath('//*[@id="amount"]')
up_button = driver.find_element_by_xpath('//*[@id="up_button"]')
down_button = driver.find_element_by_xpath('//*[@id="down_button"]')

if clickbuttontest==1:
    print 'now conducting a click buttons test...'
    makebet(1,bet_field,up_button,down_button,bet_button,betamount)
    time.sleep(10)
    terminateall(driver)

currvaltext = driver.find_element_by_xpath('//*[@id="strike"]')
currcredit = driver.find_element_by_xpath('//*[@id="balance"]')
print datetime.now()

#load AI
AIdir  = './AI/tf/'
AIname = AIdir+AIname

print 'Predictions will be made using',AIname

possibleactions = (downparam,restparam,upparam)
weightactions = np.array([0.2,0.6,0.2])

learner = learning_util.learn(AIname,0.001,possibleactions,weightactions,int(interval*60/periodint))
learner.loadmodel()

print 'main loop'

#Initialize
lastbet=-100000
countTAs = 0

qqqq = 1000
while qqqq > 0 :
    while (checktradetime()) :
        "It's too early!!!!"
        time.sleep(20*60)

    outdir = './data/realdata/'
    outnum=len([name for name in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, name))])+1

    time_list,val_list = initialization(interval,periodint,currvaltext)
    for j in range(0,int(60*maxmin/periodint)):
        currval = float(currvaltext.text)
        time_list.extend([datetime.now()])
        val_list = np.append(val_list,currval)
        pred_nows=[]
        input_now = makeinput(interval,periodint,val_list)
        #input_now = input_now.reshape(1,-1)
        input_now = data_util.scaling(input_now)
        pred_now = learner.select_action_norandom(input_now)

        if (((j-lastbet)*periodint >= waittime)  and ( (pred_now== upparam ) | (pred_now == downparam)) ):
            print 'Chance!! AI predicted value is', pred_now, \
                    'the time is now', datetime.now()
            makebet(pred_now,bet_field,up_button,down_button,bet_button,betamount)
            countTAs += 1
            lastbet=j

        if int(j*periodint) % (60) == 0:
            if ((j>0) and (int(j*periodint) % (60*30) == 0)):
                print 'outputting...'
                output_data(time_list,val_list,outnum)

            print j*periodint/60 ,'min in main loop elapsed'
            print 'we have',currcredit.text,'in the account'
            print countTAs,'transactions are made.'

        if(checktradetime()):
            #terminateall(driver)
            break

        time.sleep(periodint)

    qqqq += 1

terminateall(driver)
