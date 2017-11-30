README
=====

# Warranty
There is no guarantee that these codes works. 
I have no responsibility on any of the consequences arising from modifying,running, and distributing this code.

# How to use
Go to 

`http://www.histdata.com/download-free-forex-historical-data/?/ninjatrader/tick-last-quotes/usdjpy/`

Download the zip files for all the month from January of 2015 up to now, extract, and put the csv files in 

`tickdata/rawdata/`

Go in to the folder `shapedata` run 

`python truncatenight.py`
and 
`python shapetwosec.py`

This will generate a csv files with all the USD/JPY data in an interval of 1 second
(note that the linear extrapolation is taken for missing data points).

Enjoy.
