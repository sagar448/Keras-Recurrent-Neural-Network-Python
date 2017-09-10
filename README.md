# Keras Recurrent Neural Network With Python

<p align="center">
  <img width="460" height="300" src="https://upload.wikimedia.org/wikipedia/commons/c/c9/Keras_Logo.jpg">
</p>

Lets get straight into it, this tutorial will walk you through the steps to implement Keras with Python and thus to come up with a generative model.

So what exactly is Keras? Let's put it this way, it makes programming machine learning algorithms much much easier. It simply runs atop Tensorflow/Theano, cutting down on the coding and increasing efficiency. In more technical terms, Keras is a high-level neural network API written in Python. 

Let's get started, I am assuming you all have Tensorflow and Keras installed.

```
Note: It's very important you have enough knowledge about recurrent neural networks before beginning 
this tutorial. Please refer to these links for further info! 
```
## Implementation

### Imports
It was quite sometime after I managed to get this working, it took hours and hours of research! Error on the input data, not enough material to train with, problems with the activation function and even the output looked like an alien jumped out it's spaceship and died on my screen. Although challenging, the hard work paid off!

To make it easier for everyone, I'll break up the code into chunks and explain them individually. 

We start of by importing essential libraries...
```python
1    import numpy as np
2    from keras.models import Sequential
3    from keras.layers import Dense
4    from keras.layers import Dropout
5    from keras.layers import LSTM
6    from keras.utils import np_utils
```
**Line 1**, this is the numpy library. Used to perform mathematical functions, can be used for matrix multiplication, arrays etc. We will be using it to structure our input, output data and labels.

**Lines 1-6**, represents the various Keras library functions that will be utilised in order to construct our RNN. 
* Sequential: This essentially is used to create a linear stack of layers
* Dense: This simply put, is the output layer of any NN/RNN. It performs the output = activation(dot(input, weights) + bias)
* Dropout: RNNs are very prone to overfitting, this function ensures overfitting remains to a minimum. It does this by selecting random neurons and ignoring them during training, or in other words "dropped-out"
* LSTM: Long-Short Term Memory Unit
* np_utils: Specific tools to allow us to correctly process data and form it into the right format

Don't worry if you don't fully understand what all of these do! I will expand more on these as we go along.

### Input Data
Before we begin the actual code, we need to get our input data. My input will be a section of a play from the playwright genius Shakespeare. I will be using a monologue from Othello. You can get the text file from [here](Othello.txt)

Name it whatever you want. I'm calling mine "Othello.txt". Save it in the same directory as your Python program.

### Tools for Formatting
Although we now have our data, before we can input it into an RNN, it needs to be formatted. It needs to be what Keras identifies as input, a certain configuration.

```python
1    #Read the data, turn it into lower case
2    data = open("Othello.txt").read().lower()
3    #This get the set of characters used in the data and sorts them
4    chars = sorted(list(set(data)))
5    #Total number of characters used in the data
6    totalChars = len(data)
7    #Number of unique chars
8    numberOfUniqueChars = len(chars)
```
To implement the certain configuration we first need to create a couple of tools.

**Line 2** opens the text file in which your data is stored, reads it and converts all the characters into lowercase. Lowercasing characters is a form of normalisation. If the RNN isn't trained properly, capital letters might start popping up in the middle of words, for example "scApes".

**Line 4** creates a sorted list of characters used in the text. For example, for me it created the following:
```python
['\n', ' ', "'", ',', '-', '.', ';', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'y']
```
**Line 6** simply stores the total number of characters in the entire dataset into totalChars

**Line 8** stores the number of unique characters or the length of chars

Now we need to create a dictionary of each character so it can be easily represented.
```python
1    #This allows for characters to be represented by numbers
2    CharsForids = {char:Id for Id, char in enumerate(chars)}
3    #This is the opposite to the above
4    idsForChars = {Id:char for Id, char in enumerate(chars)}
5    #How many timesteps e.g how many characters we want to process in one go
6    numberOfCharsToLearn = 100
```
**Line 2** creates a dictionary where each character is a key. Each key character is represented by a number. For example entering this...
```python
CharsForids["o"]
```
 ...into the shell outputs..
 ```python
 20
 ```
**Line 4** is simply the opposite of **Line 2**. Now the number is the key and the corresponding character is the value. 

**Line 6** is basically how many characters we want one training example to contain or in other words the number of time-steps.

### Formatting
Our tools are ready! We can now format our data!
```python
#Input data
charX = []
#Output data
y = []
#Since our timestep sequence represetns a process for every 100 chars we omit
#the first 100 chars so the loop runs a 100 less or there will be index out of
#range
counter = totalChars - numberOfCharsToLearn
#This loops through all the characters in the data skipping the first 100
for i in range(0, counter, 1):
    #This one goes from 0-100 so it gets 100 values starting from 0 and stops
    #just before the 100th value
    theInputChars = data[i:i+numberOfCharsToLearn]
    #With no ':' you start with 0, and so you get the actual 100th value
    #Essentially, the output Chars is the next char in line for those 100 chars in charX
    theOutputChars = data[i + numberOfCharsToLearn]
    #Appends every 100 chars ids as a list into charX
    charX.append([CharsForids[char] for char in theInputChars])
    #For every 100 values there is one y value which is the output
    y.append(CharsForids[theOutputChars])
```
