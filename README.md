# Image-Labelling (Neural Network from scratch implemetation)

This code implements an optical character recognizer using a one hidden layer neural network with sigmoid activations. It learns the parameters of the model on the training data, report the cross-entropy at the end of each epoch on both train and validation data, and at the end of training write out its predictions and error rates on both datasets.

Pointers:

1. Model uses a sigmoid activation function on the hidden layer and softmax on the output layer to ensure it forms a proper probability distribution.
2. Uses stochastic gradient descent (SGD) to optimize the parameters for one hidden layer neural network.
3. This implementation is without shuffling of the data for SGD

How to Run:

Command - python neuralnet.py [args...]

Where above [args...] is a placeholder for nine command-line arguments: <train input> test input> <train out> <test out> <metrics out> <num epoch> <hidden units> <init flag> <learning rate>. These arguments are described in detail below:
1. <train input>: path to the training input .csv file
2. <test input>: path to the test input .csv file 
3. <train out>: path to output .labels file to which the prediction on the training data should be written
4. <test out>: path to output .labels file to which the prediction on the test data should be written
5. <metrics out>: path of the output .txt file to which metrics such as train and test error should be written
6. <num epoch>: integer specifying the number of times backpropogation loops through all of the
training data (e.g., if <num epoch> equals 5, then each training example will be used in backpropogation 5 times).
7. <hidden units>: positive integer specifying the number of hidden units.
8. <init flag>: integer taking value 1 or 2 that specifies whether to use RANDOM or ZERO initialization that is, if init_flag==1 initialize your weights randomly from a uniform distribution over the range [-0.1,0.1] (i.e. RANDOM), if init_flag==2 initialize all
weights to zero (i.e. ZERO). For both settings, always initialize bias terms to zero.
9. <learning rate>: float value specifying the learning rate for SGD.
