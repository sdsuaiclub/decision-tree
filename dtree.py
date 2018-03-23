# lets first get familiar with the data
from sklearn.datasets import load_iris #import the data in

iris = load_iris() #this dataset is loadded in with sci-kit
print(iris) #lets see all the data, its a dictionary made up of 2 keys, data and target


#=== lets see the data
print(iris.data[0]) # lets see the first row
print(iris.feature_names) # lets see the features 
print(iris.target[0])


# ===== now we will get the index of where each flower starts
from sklearn import tree #importing the decision tree library
test_index = [0,50,100] #this is where the data is split with 0 being setosa, 50 being where veriscolor is and 100 being where virginica is
#these index represent the first row of each type of flower
print(test_index)


#==== here we are creating the training data target labels
import numpy as np
#numpy.delete(array, object, axis = None) : returns a new array with the deletion of sub-arrays along with the mentioned axis.
train_target = np.delete(iris.target,test_index)
print(training_target)


#=== here we are pulling out the data for the training data
train_data = np.delete(iris.data, test_index, axis=0)
print(train_data)


#=== finally here is the test data - do you guys know what the difference between test and train data is?
test_data = iris.data[test_index]


#===  training the model
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target) #this is where you train your model, it will do the tree spliting based on the CART algorithm


#=== visualization code 
import graphviz 
 dot_data = tree.export_graphviz(clf, out_file=None) 
 graph = graphviz.Source(dot_data) 
 graph.render("iris")


 dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
 graph = graphviz.Source(dot_data)  
 graph 


#=== seeing how good our model is
 #print(iris.data[:50,:])
clf.predict(iris.data[50:100, :])