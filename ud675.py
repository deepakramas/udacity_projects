# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Problem 1

# <headingcell level=2>

# Below, the model is selected that has smallest generalization error amongst all models and then within a model, the complexity is selected that strikes a balance between underfitting and overfitting.

# <headingcell level=3>

# A) Decision tree

# <codecell>

mlerror = {}
mlcomplexity = {}

# <codecell>

#%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/complexity/dt.py
%pylab --no-import-all inline

# <codecell>

"""
Plots Model Complexity graphs for Decision Trees
For Decision Trees we vary complexity by changing the size of the decision tree
"""

from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

# Load the boston dataset and seperate it into training and testing set
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will vary the depth of decision trees from 2 to 25
max_depth = arange(2, 25)
train_err = zeros(len(max_depth))
test_err = zeros(len(max_depth))

for i, d in enumerate(max_depth):
	# Setup a Decision Tree Regressor so that it learns a tree with depth d
    regressor = DecisionTreeRegressor(max_depth=d)
    
    # Fit the learner to the training data
    regressor.fit(X_train, y_train)

	# Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the depth of the decision tree learnt
pl.figure()
pl.title('Decision Trees: Performance vs Max Depth')
pl.plot(max_depth, test_err, lw=2, label = 'test error')
pl.plot(max_depth, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Max Depth')
pl.ylabel('RMS Error')
pl.show()

# <headingcell level=3>

# Find minimum test error for Decision Tree and corresponding depth

# <codecell>

mlerror['decision'] = min(test_err)
mlcomplexity['decision'] = list(test_err).index(mlerror['decision'])

# <headingcell level=3>

# B) Adaboost

# <codecell>

#%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/complexity/boosting.py

# <codecell>

"""
Plots Model Complexity graphs for boosting, Adaboost in this case
For Boosting we vary model complexity by varying the number of base learners
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load the boston dataset and seperate it into training and testing set
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target)
#offset = int(0.7*len(X))
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

# We will vary the number of base learners from 2 to 300
max_learners = arange(2, 300)
train_err = zeros(len(max_learners))
test_err = zeros(len(max_learners))

for i, l in enumerate(max_learners):
	# Set up a Adaboost Regression Learner with l base learners    
    regressor = AdaBoostRegressor(n_estimators=l,random_state=0)

    # Fit the learner to the training data
    regressor.fit(X_train, y_train)

    # Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the number of base learners
pl.figure()
pl.title('Boosting: Performance vs Number of Learners')
pl.plot(max_learners, test_err, lw=2, label = 'test error')
pl.plot(max_learners, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Learners')
pl.ylabel('RMS Error')
pl.show()

# <headingcell level=3>

# Find minimum error for adaboost and corresponding max_learner

# <codecell>

mlerror['adaboost'] = min(test_err)
mlcomplexity['adaboost'] = list(test_err).index(mlerror['adaboost'])

# <headingcell level=2>

# C) kNN

# <codecell>

#%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/complexity/knn.py

# <codecell>

"""
Plots Model Complexity graphs for kNN
For kNN we vary complexity by chaning k
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load the boston dataset and seperate it into training and testing set
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target)
#offset = int(0.7*len(X))
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

# We will change k from 1 to 30
k_range = arange(1, 30)
train_err = zeros(len(k_range))
test_err = zeros(len(k_range))

for i, k in enumerate(k_range):
	# Set up a KNN model that regressors over k neighbors
    neigh = KNeighborsRegressor(n_neighbors=k)
    
    # Fit the learner to the training data
    neigh.fit(X_train, y_train)

	# Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, neigh.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, neigh.predict(X_test))

# Plot training and test error as a function of k
pl.figure()
pl.title('kNN: Error as a function of k')
pl.plot(k_range, test_err, lw=2, label = 'test error')
pl.plot(k_range, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('k')
pl.ylabel('RMS error')
pl.show()

# <headingcell level=3>

# Find minimum test error and corresponding complexity for kNN

# <codecell>

mlerror['knn'] = min(test_err)
mlcomplexity['knn'] = list(test_err).index(mlerror['knn'])

# <headingcell level=2>

# D) Neural Network

# <codecell>

#%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/complexity/nn.py

# <codecell>

"""
Plots Performance of Neural Networks when you change the network
We vary complexity by changing the number of hidden layers the network has
We use pybrain (http://pybrain.org/) to design and train our NN
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# Load the boston dataset and seperate it into training and testing set
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target)
#offset = int(0.7*len(X))
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

# List all the different networks we want to test again
# All networks have 13 input nodes and 1 output nodes
# All networks are fully connected
net = []
# 1 hidden layer with 1 node
net.append(buildNetwork(13,1,1))
# 1 hidden layer with 5 nodes
net.append(buildNetwork(13,5,1))
# 2 hidden layers with 7 and 3 nodes resp
net.append(buildNetwork(13,7,3,1))
# 3 hidden layers with 9, 7 and 3 nodes resp
net.append(buildNetwork(13,9,7,3,1))
# 4 hidden layers with 9, 7, 3 and 2 noes resp
net.append(buildNetwork(13,9,7,3,2,1))
net_arr = range(0, len(net))

# The dataset will have 13 features and 1 target label
ds = SupervisedDataSet(13, 1)

train_err = zeros(len(net))
test_err = zeros(len(net))

# We will train each NN for 50 epochs
max_epochs = 50

# Convert the boston dataset into SupervisedDataset
for j in range(1, len(X_train)):
	ds.addSample(X_train[j], y_train[j])

for i in range(1, len(net)):
	# Setup a trainer that will use backpropogation for training
	trainer = BackpropTrainer(net[i], ds)

	# Run backprop for max_epochs number of times
	for k in range(1, max_epochs):
		train_err[i] = trainer.train()

	# Find the labels for test set
	y = zeros(len(X_test))

	for j in range(0, len(X_test)):
		y[j] = net[i].activate(X_test[j])

    # Calculate MSE for all samples in the test set
	test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the number of hidden layers
pl.figure()
pl.title('Neural Networks: Performance vs Model Complexity')
pl.plot(net_arr, test_err, lw=2, label = 'test error')
pl.plot(net_arr, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Model Complexity')
pl.ylabel('RMS Error')
pl.show()

# <headingcell level=3>

# Find minimum test error and corresponding complexity for NN

# <codecell>

mlerror['nn'] = min(test_err[1:])
mlcomplexity['nn'] = list(test_err).index(mlerror['nn'])

# <headingcell level=2>

# Inspecting the curves, boosting is selected as a model since it has the lowest generalization error. 

# <headingcell level=2>

# Boosting Model of complexity = 100 is selected as after this point overfitting appears to occur and prior to this value underfitting appears to occur.

# <codecell>

X_problem = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13], ndmin=2)
# Note the use of random_state to reproduce the boosting regressor seen earlier for this complexity 
regressor = AdaBoostRegressor(n_estimators=100,random_state=0)
regressor.fit(X_train, y_train)
Y_problem = regressor.predict(X_problem)
print Y_problem

# <headingcell level=2>

# Predicted price is 20.59

# <codecell>

Y_problems = zeros(len(max_learners))
for i, l in enumerate(max_learners):
	# Setup a Decision Tree Regressor so that it learns a tree with depth d
    regressor = AdaBoostRegressor(n_estimators=l,random_state=0)
    
    # Fit the learner to the training data
    regressor.fit(X_train, y_train)

    Y_problems[i] = regressor.predict(X_problem)

print np.var(Y_problems)

# <headingcell level=2>

# Variance across complexity is 0.0357

# <headingcell level=1>

# Problem 2

# <headingcell level=2>

# All the models are provided the same data to learn and are independently evaluated on different data called test data that is same for all models. Our model provides the best performance amongst all models and strikes a balance between underfitting and overfitting the data.

# <headingcell level=1>

# Problem 3a

# <headingcell level=2>

# Decision Trees: The gap between the test and training errors isn't too large at ~15 compared to the other models. This shows that the model has reasonable generalization error. The best model occurs for depth=5

# <headingcell level=2>

# Boosting: The gap between test and training curves is the smallest. Therefore this model has the best generalization. The best model is for complexity=100

# <headingcell level=2>

# kNN: At k=3, the lowest test error is achieved. While the test error is high at ~37, the gap between the test and train curves at this point is ~15 making it comparable to DT. The model thus ties with DT in having 2nd best generalization. 

# <headingcell level=2>

# NN: The gap between test and train curves is the highest at ~20 with high test error as well. NN therefore has the worst generalization amongst all models. The best model is for complexity=2

# <headingcell level=1>

# Problem 3b

# <codecell>

%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/learning_curves/dt.py

# <codecell>

"""
Plots Learning curves for Decision Trees
Plots performance of DecisionTreeRegressor when we change the size of the training set
"""

from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

# Load the boston dataset and seperate it into training and testing set
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target)
#offset = int(0.7*len(X))
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

# We will vary the training set size in increments of 20
sizes = linspace(1, len(X_train), 20)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

for i, s in enumerate(sizes):
	# Fit a model where the maximum depth of generated decision tree is 5
    regressor = DecisionTreeRegressor(max_depth=5)
    regressor.fit(X_train[:s], y_train[:s])

	# Find the MSE on the training and testing set
    train_err[i] = mean_squared_error(y_train[:s], regressor.predict(X_train[:s]))
    test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the training size
pl.figure()
pl.title('Decision Trees: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()


# <headingcell level=2>

# Decision Tree: The curves show that DT are low bias and low variance estimators. Adding more data is worth while as the model will generalize better.

# <codecell>

#%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/learning_curves/boosting.py

# <codecell>

"""
Plots Learning curves for Boosting
Plots performance of an AdaBoostRegressor when we change the size of training set
"""

from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.ensemble import AdaBoostRegressor

# Load the boston dataset and seperate it into training and testing set
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target)
#offset = int(0.7*len(X))
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

# We will vary the training set size in increments of 20
sizes = linspace(1, len(X_train), 20)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

for i, s in enumerate(sizes):
	# Fit a model with 100 base learners
	regressor = AdaBoostRegressor(n_estimators=100, random_state=0)
	regressor.fit(X_train[:s], y_train[:s])

	# Find the MSE on the training and testing set
	train_err[i] = mean_squared_error(y_train[:s], regressor.predict(X_train[:s]))
	test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the training size
pl.figure()
pl.title('Boosting: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()


# <headingcell level=2>

# Boosting: The curves show boosting are low bias and low variance estimators. In fact boosting provides the lowest bias and variance amongst all estimators. Adding more data after N=250, is unnecessary as the model has generalized very well.

# <codecell>

#%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/learning_curves/knn.py

# <codecell>

"""
Plots Learning curves for kNN
Plot performance of kNN when we change the size of the training set
"""

from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor

# Load the boston dataset and seperate it into training and testing set
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target)
#offset = int(0.7*len(X))
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

# We will vary the training set size in increments of 20
sizes = linspace(1, len(X_train), 20)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

for i, s in enumerate(sizes):
	# Fit a kNN model with k=3
	regressor = KNeighborsRegressor(n_neighbors=3)
	regressor.fit(X_train[:s], y_train[:s])

	# Find the MSE on the training and testing set
	train_err[i] = mean_squared_error(y_train[:s], regressor.predict(X_train[:s]))
	test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the training size
pl.figure()
pl.title('3-NN: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()


# <headingcell level=2>

# kNN: The test curve shows that the variance is large and the bias is not too small either. Increasing the size of the data will not reduce either the variance or the bias. The model does not generalize well.

# <codecell>

%load https://raw.githubusercontent.com/pushkar/ud675-proj/master/learning_curves/nn.py

# <codecell>

-labels for the test set
    y = zeros(len(X_test))
    for j in range(0, len(X_test)):
        y[j] = net.activate(X_test[j])

    # Find MSE for the test set
    test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the training size
pl.figure()
pl.title('Neural Networks: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'trainin"""
Plots Learning curves for Neural Networks
Plot performance of NN when we change the size of the training set
"""

import sys
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# Load the boston dataset and seperate it into training and testing set
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target)
#offset = int(0.7*len(X))
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

# We will vary the training set so that we have 10 different sizes
sizes = linspace(10, len(X_train), 10)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

# Build a network with 2 hidden layers
net = buildNetwork(13, 7, 3, 1)
# The dataset will have 13 input features and 1 output
ds = SupervisedDataSet(13, 1)

for i,s in enumerate(sizes):
    # Populate the dataset for training
    ds.clear()
    for j in range(1, int(s)):
        ds.addSample(X_train[j], y_train[j])
-
    # Setup a backprop trainer
    trainer = BackpropTrainer(net, ds)

    # Train the NN for 50 epochs
    # The .train() function returns MSE over the training set
    for e in range(0, 50):
        train_err[i] = trainer.train()

    # Find g error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()

# <headingcell level=2>

# NN: Based on where the test and training curves appear to meet, NN is high bias and high variance estimator. Based on the slope of the training curve, the variance is sensitive to the size of the data. Therefore with additional data, model will generalize better. 

# <codecell>


