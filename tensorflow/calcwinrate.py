#!/usr/bin/python
from __future__ import division
initmoney=200
asset=803.0
bet=20
rate=0.85
NTA=375

x = (-asset + initmoney + bet*rate*NTA)/(bet*(rate+1))
rate = 1-x/NTA

print 'Out of',NTA,'transactions'
print 'Number of wins are', NTA-x
print 'winning rate is',rate,'%'

