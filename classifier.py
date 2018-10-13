from sklearn import neighbors
from sklearn import tree as treeClassifier
from sklearn import svm as svmClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from metrics import false_rejection_rate, false_acceptance_rate

def knn(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, k = 1, weight='uniform'):
    clf = neighbors.KNeighborsClassifier(k, weights=weight)
    name = "KNN by " + weight
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name)



def tree(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random):
    clf = treeClassifier.DecisionTreeClassifier()
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name="Tree")
    

def svm(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, weights = None):
    clf = svmClassifier.SVC(probability=True, class_weight=weights, kernel="linear")
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name="SVM")
    

def mlp(data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1):
    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    return execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name="MLP")
    

def execute_test(clf, data_train, data_test, expected, correct_class, number_of_genuine, number_of_skilled, number_of_random, name):
    clf.fit(data_train, expected)
    prediction_probability = clf.predict_proba(data_test)
    scores = prediction_probability[:, 1]
    fpr, tpr, thresholds = roc_curve(correct_class, scores)
    auc_metric = auc(fpr, tpr)
    frr, far_skilled, far_random, eer = [None,None,None,None]
    for threshold in prediction_probability:
        prediction = []
        for pred in prediction_probability:
            if pred[1] <= threshold[1]:
                prediction.append(0)
            else:
                prediction.append(1)
        fn = 0
        only_genuine_prediction = prediction[:number_of_genuine]
        for pred in only_genuine_prediction:
            if pred == 0:
                fn += 1 
        frr, far_skilled, far_random = __classification_metrics(prediction, number_of_genuine, number_of_skilled, number_of_random)
        if(far_skilled == frr):
            eer = frr
            break
    prediction_global = clf.predict(data_test)
    prediction_global = prediction_global.tolist()
    frr_global, far_skilled_global, far_random_global = __classification_metrics(prediction_global, number_of_genuine, number_of_skilled, number_of_random)
    eer_global = None
    if(far_skilled_global == frr_global):
            eer_global = frr_global
    return [frr, far_skilled, far_random, eer, frr_global, far_skilled_global, far_random_global, eer_global, auc_metric]

def __classification_metrics(prediction, number_of_genuine, number_of_skilled, number_of_random):
    fn = 0
    only_genuine_prediction = prediction[:number_of_genuine]
    for pred in only_genuine_prediction:
        if pred == 0:
            fn += 1 
    frr = false_rejection_rate(number_of_genuine, fn)
    only_forgery_prediction = prediction[number_of_genuine:]
    skilled_prediction = only_forgery_prediction[:number_of_skilled]
    random_prediction = only_forgery_prediction[number_of_skilled:]
    fp_skilled =  skilled_prediction.count(1)
    fp_random = random_prediction.count(1)
    far_skilled = false_acceptance_rate(number_of_skilled, fp_skilled)
    far_random = false_acceptance_rate(number_of_random, fp_random)
    return [frr, far_skilled, far_random]