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
1     #Input data
2     charX = []
3     #Output data
4     y = []
5     #Since our timestep sequence represetns a process for every 100 chars we omit
6     #the first 100 chars so the loop runs a 100 less or there will be index out of
7     #range
8     counter = totalChars - numberOfCharsToLearn
9     #This loops through all the characters in the data skipping the first 100
10    for i in range(0, counter, 1):
11        #This one goes from 0-100 so it gets 100 values starting from 0 and stops
12        #just before the 100th value
13        theInputChars = data[i:i+numberOfCharsToLearn]
14        #With no ':' you start with 0, and so you get the actual 100th value
15        #Essentially, the output Chars is the next char in line for those 100 chars in charX
16        theOutputChars = data[i + numberOfCharsToLearn]
17        #Appends every 100 chars ids as a list into charX
18        charX.append([CharsForids[char] for char in theInputChars])
19        #For every 100 values there is one y value which is the output
20        y.append(CharsForids[theOutputChars])
```
**Line 2, 4** are empty lists for storing the formatted data as input, charX and output, y

**Line 8** creates a counter for our for loop. We run our loop for a 100 (numberOfCharsToLearn) less as we will be referencing the last 100 as the output chars or the consecutive chars to the input

**Line 13** theInputChars stores the first 100 chars and then as the loop iterates, it takes the next 100 and so on...

**Line 16** theOutputChars stores only 1 char, the next char after the last char in theInputChars

**Line 18** the charX list is appended to with 100 integers. Each of those integers are IDs of the chars in theInputChars

**Line20** appends an integer ID every iteration to the y list corresponding to the single char in theOutputChars

Are we now ready to put our data through the RNN? Not quite! We have the data represented correctly but still not in the right format
```python
1    #Len(charX) represents how many of those time steps we have
2    #The numberOfCharsToLearn is how many character we process
3    #Our features are set to 1 because in the output we are only predicting 1 char
4    X = np.reshape(charX, (len(charX), numberOfCharsToLearn, 1))
5    #This is done for normalization
6    X = X/float(numberOfUniqueChars)
7    #This sets it up for us so we can have a categorical(#feature) output format
8    y = np_utils.to_categorical(y)
```
**Line 4** shapes the input array into [samples, time-steps, features], required for Keras

**Line 6** this is a form of normalisation

**Line 8** this converts y into a one-hot vector. A one-hot vector is an array of 0s and 1s. The 1 only occurs at the position where the ID is true. For example, say we have 5 unique character IDs, [0, 1, 2, 3, 4]. Then say we have 1 single data output equal to 1, y = ([[0, 1, 0, 0, 0]]). Notice how the 1 only occurs at the position of 1. Now imagine exactly this, but for 100 different examples with a length of numberOfUniqueChars

### Building the RNN model
Thats data formatting and representation part finished!
Yes! We can now start building our RNN model!
```python
model = Sequential()
#Since we know the shape of our Data we can input the timestep and feature data
#The number of timestep sequence are dealt with in the fit function
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
#number of features on the output
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=5, batch_size=128)
model.save_weights("Othello.hdf5")
#model.load_weights("Othello.hdf5")
```
**Line 1** this uses the Sequential() import I mentioned earlier.
This essentially initialises the network. It creates an empty "template model".

**Line 6** we now add our first layer to the empty "template model". This is the LSTM layer which contains 256 LSTM units, with the input shape being input_shape=(numberOfCharsToLearn, features). It was written that way to avoid any silly mistakes! Although the X array is of 3 dimensions we omit the "samples dimension" in the LSTM layer because it is accounted for automatically later on.
```
Note: Omitting does not mean the "samples dimension" is not considered!
```
**Line 7** this as explained in the imports section "drops-out" a neuron. The 0.2 represents a percentage, it means 20% of the neurons will be "dropped" or set to 0

**Line 9*** the layer acts as an output layer. It performs the activation of the dot of the weights and the inputs plus the bias
```
Note: RNNs do not actually utilise the activation function in its recurrent components to minimise the vanishing gradient problem!
```
**Line 10** this is the configuration settings. Our loss function is the "categorical_crossentropy" and the optimizer is "Adam"

**Line 11** runs the training algorithm. The epochs are the number of times we want each of our batches to be evaluated. I have set it to 5 for this tutorial but generally 20 or higher epochs are favourable. The batch size is the how many of our input data set we want evaluated at once. In this case we input 128 of examples into the training algorithm then the next 128 and so on..

**Line 12**, finally once the training is done, we can save the weights

**Line 13** this is commented out initially to prevent errors but once we have saved our weights we can comment out **Line 11, 12** and uncomment **line 13** to load previously trained weights
