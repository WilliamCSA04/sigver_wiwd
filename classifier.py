from sklearn import neighbors
from sklearn import tree as treeClassifier
from sklearn import svm as svmClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from metrics import false_rejection_rate, false_acceptance_rate, threshold

def knn(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, k = 1, weight='uniform'):
    print("KNN Classifier")
    clf = neighbors.KNeighborsClassifier(k, weights=weight)
    name = "KNN by " + weight
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name)



def tree(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random):
    clf = treeClassifier.DecisionTreeClassifier()
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name="Tree")
    

def svm(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random):
    clf = svmClassifier.SVC()
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name="SVM")
    

def mlp(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1):
    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name="MLP")
    

def execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name):
    clf.fit(data_train, expected)
    prediction = clf.predict(data_test)
    print("Prediction: " + name) 
    print(prediction)
    acc = accuracy_score(correct_class, prediction)
    print("Acc: " + str(acc))
     
    tn, fp, fn, tp = confusion_matrix(correct_class, prediction).ravel()
    frr = false_rejection_rate(number_of_genuine, fn)
    only_forgery_prediction = prediction[number_of_genuine:]
    skilled_prediction = only_forgery_prediction[:number_of_skilled]
    random_prediction = only_forgery_prediction[number_of_skilled:]
    fp_skilled =  skilled_prediction.tolist().count(1)
    fp_random = random_prediction.tolist().count(1)
    far_skilled = false_acceptance_rate(number_of_skilled, fp_skilled)
    far_random = false_acceptance_rate(number_of_random, fp_random)
    trh = threshold(far_skilled, frr)
    print("frr: ")
    print(frr)
    print("far_skilled: ")
    print(far_skilled)
    print("far_random: ")
    print(far_random)
    return [frr, far_skilled, far_random, trh]    