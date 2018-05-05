README
=====

# BETA VERSION IS COMPLETED!
# Current AI in the master node achieves the win rate of 65% in the backtesting on 2017 data!

# PLEASE HELP! PLEASE CONTRIBUTE!

# Warranty
There is no guarantee that these codes works. 
I have no responsibility on any of the consequences arising from modifying,running, and distributing this code.

# Warning

There is nothing amazing happening right now. It is under development, and there is no actual profits or real-time signaling properties
so far. It will happen.

There is no GUI because I have no idea how to do that.

This is not for beginners because I'm not expert enough to make it a beginner friendly... I'm so sorry...

# Dependencies
It requires `python 2.7` and packages that can be installed with `pip` (I'll make an official list). 
I'm a python noob, so I don't know when it will be. It would be nice if I can make some `setup.py`.

# Progress
If you backtest using the AI in Master node, you can see that the winrate is around 65% (plus or minus 3%)!

# TO DO
+ 日本語のドキュメンテーションの作成
+ ~~Implimentation of RMSPropGraves~~ This will never happen.

# Note

Please report any issues or questions on `issues` tab, so that I can ask all the questions at once, 
and your question might already be answered there.


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

## Restart from the saved AI
Go to tensorflow folder,
and run `makeAI-tf.py`, but this time make sure to turn `cont_learn` to `= True` at line `23`
if you want to continue from the AI that is made previously.

## Backtesting

Backtest can be done using `tensorflow/backtest.py`. Just run it and it should output how much you have profitted 
if you traded in whatever years you specified.

**The key to the success is a corrrect Qthresh value! right now it is 1.45,
but if you train further you might want to increase it!**

## Trying it on demotrade

You can run `demotrade_tf.py` to test the AI on the real trading website (it is a demo trade from high-low Australia).

Enjoy.

### sklearn version is obsolete. 
run 
`make-AI-skl.py` this will create a base AI
then run
`prob_clas.py` this takes a probability distribution of the base AI and make further predictions.
then run
`backtest.py` to conduct backtesting.


## Contact
If you have any questions or bugs, either submit in the issues tab or directly contact me via e-mail
(t.hashizume.brisbane@gmail.com).


<!---
## Donation
If you can't technically contribute but want to help, please send some bitcoins to
Any amount will help my studies, which means more free time, and more time to edit and tryout different techniques.
--->

