import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
#print(X),k
deno = np.amax(X,axis=0)
print("Demo",deno)
X = X/deno
y = y/100
print(X)
print(y)

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivatives_sigmoid(x):
    return x * (1 - x)

iteration = 100	#Setting training iterations
error_rate =0.1 		#Setting learning rate-error rate
inputlayer_neurons = 2 		#number of features in data set
hiddenlayer_neurons = 3 	#number of hidden layers neurons
output_neurons = 1 		#number of neurons at output layer
weight_hidden = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
#print(weight_hidden)
bias_hidden = np.random.uniform(size=(1,hiddenlayer_neurons))
#print("bias value",bias_hidden)
weight_output = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bias_output = np.random.uniform(size=(1,output_neurons))
print(X.shape)

for i in range(iteration):
    hidden_input=np.dot(X,weight_hidden) # h1 = w1*x1+w2*x2+w3*x3
    hidden_input =hidden_input + bias_hidden
    output_hidden_layer = sigmoid( hidden_input)
    #print(output_hidden_layer)
    actual_output = np.dot(output_hidden_layer,weight_output)+bias_output
    y_output = sigmoid(actual_output)
    #print("predicted output",y_output)


    #Backpropagation
    EO = y-y_output
    outgrad = derivatives_sigmoid(y_output)
    d_output = EO* outgrad
    EH = d_output.dot(weight_output.T)
    hiddengrad = derivatives_sigmoid(output_hidden_layer)
    d_hiddenlayer = EH * hiddengrad

    weight_output += output_hidden_layer.T.dot(d_output) *error_rate
    weight_hidden += X.T.dot(d_hiddenlayer) *error_rate

print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,y_output)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("Advertising.csv")
data.head()
data.drop(['Unnamed: 0'], axis=1)

plt.figure(figsize=(16, 8))
plt.scatter(
 data['TV'],
 data['sales'],
 c='black'
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()
X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
reg = LinearRegression()
reg.fit(x_train, y_train)
print("Slope: ",reg.coef_[0][0])
print("Intercept: ",reg.intercept_[0])
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(x_test)
plt.figure(figsize=(16, 8))
plt.scatter(
 x_test,
 y_test,
 c='black'
)
plt.plot(
 x_test,
 predictions,
 c='blue',
 linewidth=2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()
rmse = np.sqrt(mean_squared_error(y_test,predictions))
print("Root Mean Squared Error = ",rmse)
r2 = r2_score(y_test,predictions)
print("R2 = ",r2)

import pandas as pd
import numpy as np
import math
import operator
data = pd.read_csv("iris.csv")
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
#data = pd.read_csv(url, names=names)
data.head()
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)

def knn(trainingSet, testInstance, k):
  distances = {}
  sort = {}
  length = testInstance.shape[1]
#### Start of STEP 3
# Calculating euclidean distance between each row of training data and test
  for x in range(len(trainingSet)):
#### Start of STEP 3.1
    dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
    distances[x] = dist[0]
#### End of STEP 3.1
#### Start of STEP 3.2
# Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))
#### End of STEP 3.2
    neighbors = []
#### Start of STEP 3.3
# Extracting top k neighbors
  for x in range(k):
    neighbors.append(sorted_d[x][0])
#### End of STEP 3.3
    classVotes = {}
#### Start of STEP 3.4
# Calculating the most freq class in the neighbors
  for x in range(len(neighbors)):
    response = trainingSet.iloc[neighbors[x]][-1]
    if response in classVotes:
      classVotes[response] += 1
    else:
      classVotes[response] = 1
#### End of STEP 3.4
#### Start of STEP 3.5
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1))
    return(sortedVotes[0][0], neighbors)
#### End of STEP 3.5

testSet = [[7.2, 3.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)
k = 1
#### End of STEP 2
# Running KNN model
result,neigh = knn(data, test, k)
# Predicted class
print(result)
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#import the dataset
df = pd.read_csv('iris.csv')
df.head(10)
x = df.iloc[:, [0,1,2,3]].values
kmeans5 = KMeans(n_clusters=2)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)

kmeans5.cluster_centers_
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans5.fit_predict(x)
print(y_kmeans3)

kmeans5.cluster_centers_
import pandas as pd
import numpy as np
#Generate a dummy dataset.
X = np.random.randint(10,50,100).reshape(20,5)
print(X)
# mean Centering the data
X_meaned = X - np.mean(X , axis = 0)
print(X_meaned)
# calculating the covariance matrix of the mean-centered data.
cov_mat = np.cov(X_meaned , rowvar = False)
print(cov_mat)
#Calculating Eigenvalues and Eigenvectors of the covariance matrix
eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
print("Eigen Values are", eigen_values)
print("Eigen Vectors are", eigen_vectors)
sorted_index = np.argsort(eigen_values)[::-1]
print("The Sorted Eigen Values are", sorted_index)
sorted_eigenvalue = eigen_values[sorted_index]
#similarly sort the eigenvectors
sorted_eigenvectors = eigen_vectors[:,sorted_index]
print("The Sorted Eigen Vectors are", sorted_eigenvectors)

n_components = 2 #you can select any number of components.
eigenvector_subset = sorted_eigenvectors[:,0:n_components]
print(eigenvector_subset)
#Transform the data
X_reduced = np.dot(eigenvector_subset.transpose(),X.transpose()).transpose()
print(X_reduced)
