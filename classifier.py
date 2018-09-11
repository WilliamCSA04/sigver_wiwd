from sklearn import neighbors
from sklearn import tree as treeClassifier
from sklearn import svm as svmClassifier
from sklearn.neural_network import MLPClassifier

def knn(data_train, data_test, expected, k = 1, weigths=['uniform', 'distance']):
    print("KNN Classifier")
    for weight in weigths:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(k, weights=weight)
        clf.fit(data_train, expected)
        prediction = clf.predict(data_test)
        print("Prediction KNN: " + weight) 
        print(prediction) 

def tree(data_train, data_test, expected):
    clf = treeClassifier.DecisionTreeClassifier()
    clf = clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction Tree: ") 
    print(prediction)

def svm(data_train, data_test, expected):
    clf = svmClassifier.SVC()
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction SVM: ") 
    print(prediction)

def mlp(data_train, data_test, expected, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1):
    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction MLP: ") 
    print(prediction)