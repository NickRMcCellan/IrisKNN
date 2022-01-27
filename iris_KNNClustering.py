import matplotlib.pyplot as plt

import numpy as np
from numpy import mean
from numpy import std

import scipy.io
from scipy.stats import sem

import sklearn
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#Function Definition for use later
def evaluate_model(X, t, k):
    cv = KFold(n_splits=5)
	# create model
    model = KNeighborsClassifier(n_neighbors=k)
	# evaluate model
    scores = cross_val_score(model, X, t, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

#Get data from iris dataset-------------------------------------------------
from sklearn import datasets
iris = datasets.load_iris() 
X = np.array(iris.data)
T = np.array(iris.target)

#Visualize Data with PCA
Xm= (X-np.mean(X,axis=0)).T
U, S, VT = np.linalg.svd(Xm,full_matrices=0)
L=U[:,0:2].T @ Xm

plt.figure()
plt.plot(L[0,:49],L[1,:49],'d',color='k',label='Flower 1')
plt.plot(L[0,49:99],L[1,49:99],'d',color='r',label='Flower 2')
plt.plot(L[0,99:149],L[1,99:149],'d',color='b',label='Flower 3')
plt.legend()
plt.show() 

#Building the classifier---------------------------------------------------
#We will scale the data first by mean subtracting and dividing by the standard deviation of X
Sd = np.std(X,axis=0)
Xs=Xm/Sd.T[:,np.newaxis]


#Allocate Testing and Training Data
X_train, X_test, T_train, T_test = train_test_split(Xs.T,T,test_size=.3,random_state=42)

# Cross Validates data with varying number of Neighboors
neighbors = range(3,20)
results = list()

for r in neighbors:
	# evaluate using a given number of repeats
	scores = evaluate_model(X_train, T_train, r)
	# summarize
	print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
	# store
	results.append(1-mean(scores))

plt.figure()
plt.plot(neighbors,results)
plt.show()


# Final Model and confusion matrix:
classifier = KNeighborsClassifier(n_neighbors=8).fit(X_train, T_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, T_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()





















