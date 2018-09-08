from sklearn import neighbors
from sklearn import tree
from sklearn import svm

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
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction Tree: ") 
    print(prediction)

def svm(data_train, data_test, expected):
    clf = svm.SVC()
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction SVM: ") 
    print(prediction)
