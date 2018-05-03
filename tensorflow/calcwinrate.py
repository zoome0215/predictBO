from __future__ import division
initmoney=200
asset=789.0
bet=20
rate=0.85
NTA=150

x = (-asset + initmoney + bet*rate*NTA)/(bet*(rate+1))
rate = 1-x/NTA

print 'winning rate is',rate,'%'

