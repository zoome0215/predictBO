from __future__ import division
initmoney=200
asset=2170
bet=20
rate=0.85
NTA=598

x = (-asset + initmoney + bet*rate*NTA)/(bet*(rate+1))
rate = 1-x/NTA

print 'Number of wins are', NTA-x
print 'winning rate is',rate,'%'

