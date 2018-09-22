from sklearn import neighbors
from sklearn import tree as treeClassifier
from sklearn import svm as svmClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def knn(data_train, data_test, expected, correct_class, k = 3, weights=['uniform', 'distance']):
    print("KNN Classifier")
    accs = []
    
    for index, weight in enumerate(weights):
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(k, weights=weight)
        clf.fit(data_train, expected)
        prediction = clf.predict(data_test)
        print("Prediction KNN: " + weight) 
        print(prediction) 
        acc = accuracy_score(correct_class, prediction)
        print("Acc: " + str(acc))
        accs.append(acc) 
    return accs


def tree(data_train, data_test, expected, correct_class):
    clf = treeClassifier.DecisionTreeClassifier()
    clf = clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction Tree: ") 
    print(prediction)
    print("Acc:") 
    print(accuracy_score(correct_class, prediction))

def svm(data_train, data_test, expected, correct_class):
    clf = svmClassifier.SVC()
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction SVM: ") 
    print(prediction)
    print("Acc:") 
    print(accuracy_score(correct_class, prediction))

def mlp(data_train, data_test, expected, correct_class, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1):
    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction MLP: ") 
    print(prediction)
    print("Acc:") 
    print(accuracy_score(correct_class, prediction))