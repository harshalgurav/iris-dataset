**Iris Neural Network**

This project builds a neural network with n hidden layers to predict class labels for the Iris plant dataset.

The data is split into a training and testing set. I choose to use the sigmoid activation function.

Forward and Backwards propogation is implemented from scratch and the accuracies on the training/test data sets are reported.

#IRIS-DATASET-ANALYSIS-USING-NEURAL-NETWORK

Neural Network with functions for forward propagation, error calculation and back propagation is built from scratch and is used to analyse the IRIS dataset.

#Libraries used:
->pandas for reading the input .csv file
->sklearn.preprocessing for normalising the data
->numpy for calculating exponential(x) for sigmoid function
->matplotlib for plots
->seaborn for visualising the Bivariate Pairwise relationships between features

#Inputs:
->IRIS_TrainData.csv file containning the training data

#Outputs:
->Plot display the Bivariate Pairwise relationships between features of dataset
->Plot to visualise the input values before and after normalisation
->Cost vs epoch for the trainning data using the NeuralNetwork classifier built
->Predicted output values and the species associated for the test data

Iris setosa is encoded as 0.0001
Iris versicolor is encoded as 0.9999

Normalized the features by removing the mean and scaling to unit variance using the StandardScaler module from numpy.preprocessing library
Test data inputs are normalised accordingly using the same

#User Defined functions: class NeuralNet:

#1.init(self,x,y,lr,epoch)
Inputs : x , y , lr , epoch
x=Input to NeuralNetwork
y=Target output
lr=Learning rate
epoch=epoch specified
Initialises the weight of the layers accordingly for the neural network using np.random function

#2._sigmoid(x)
Input : x
Computes and return the sigmoid(x) value

#3._sigmoid_derivative(x)
Input : x
Computes and returns the derivative of the sigmoid function i.e x(1-x)

#4.cost(y_target,y_output)
Inputs : y_target , y_output
Computes and returns the value of 0.5* sum of squares of the difference of y_target and y_output

#5.feedforward(self)
Calculates the predicted output of the layers by using the activation function

#6.backpropogation(self)
Computes the derivative of the cost function with respect to the weights and updates the weights accordingly

#7.train(self)
Implements the sequence of the steps in NeuralNetwork
self.feedforward()
self.backprop()
Also updates the costlist with cost calculated after each epoch

#8.predict(input_data)
Input : input_data for which output need to be predicted
Predicts the output for the input_data by calling the feedforward function
