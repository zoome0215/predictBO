README
=====

# Warranty
There is no guarantee that these codes works. 
I have no responsibility on any of the consequences arising from modifying,running, and distributing this code.

# Warning

There is nothing amazing happening right now. It is under development, and there is no actual profits or real-time signaling properties
so far. It will happen.

There is no GUI cuz I have no idea how to do that.

# Dependencies
`python 2.7` and packages that can be installed with `pip` (I'll make an official list). 
I'm a python noob, so I don't know when it will be. It would be nice if I can make some `setup.py`.

# How to use

## Shaping data

Go to 

`http://www.histdata.com/download-free-forex-historical-data/?/ninjatrader/tick-last-quotes/usdjpy/`

Download the zip files for all the month from January of 2014 up to now, extract, and put the csv files in 

`./data/tickdata/rawdata/`

Go in to the folder `shapedata` run 

`python truncatenight.py`
and 
`python shapetwosec.py`

This will generate a csv files with all the USD/JPY data in an interval of 1 second
(note that the linear extrapolation is taken for missing data points).

## Making AI
Go to tensorflow folder,
and run `makeAI-tf.py`. make sure to turn `cont_learn` to `= False` at line `23` if you are running it for the first time.

Once `saved!` is displayed on a terminal (the command window), run `backtest.py` and see how it goes.
tweak `checkQ` to be `= True` to see how Q-values changes over transactions as a graph.

you need to have this trained at least 100,000 times to see a good progress (still not enough, I think). 

Enjoy.

### sklearn version is obsolete. 
run 
`make-AI-skl.py` this will create a base AI
then run
`prob_clas.py` this takes a probability distribution of the base AI and make further predictions.
then run
`backtest.py` to conduct backtesting.


## Donation
If you cna't technically contribute but want to help, please send some bitcoins to


Any amount will help my studies, which means more free time, and more time to edit and tryout different techniques.


