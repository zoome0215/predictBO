#!/usr/bin/python
import os
import time

e = 0
while (e< 1000000000) :
    os.system("./backtest.py")
    time.sleep(60*60)

