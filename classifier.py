from sklearn import neighbors
from sklearn import tree as treeClassifier
from sklearn import svm as svmClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from metrics import false_rejection_rate, false_acceptance_rate, equal_error_rate_with_verification, equal_error_rate
import sys

def knn(data_train, train_classes, k = 1, weight='uniform'):
    clf = neighbors.KNeighborsClassifier(k, weights=weight)
    return clf.fit(data_train, train_classes)



def tree(data_train, train_classes):
    clf = treeClassifier.DecisionTreeClassifier()
    return clf.fit(data_train, train_classes)
    

def svm(gamma='auto', weights = None, kernel="linear"):
    return svmClassifier.SVC(probability=True, class_weight = weights, kernel = kernel, gamma = gamma)
    

def mlp(data_train, train_classes, solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1):
    clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    return clf.fit(data_train, train_classes)
    

def test(clf, test_sets, test_classes, number_of_genuine, number_of_skilled, number_of_random, global_threshold):
    results = [[], [], [], [], [], [], [], [], [], []]    
    for test_set in test_sets:
        prediction_probability = clf.predict_proba(test_set)
        scores = prediction_probability[:, 1]
        fpr, tpr, thresholds = roc_curve(test_classes, scores)
        list_of_thresholds = scores.tolist()
        auc_metric = auc(fpr, tpr)
        diff = sys.maxint
        frr_user, far_skilled_user, far_random_user, eer_user = 0, 0, 0, 0
        for threshold in list_of_thresholds:
            prediction = __prediction_list(threshold, list_of_thresholds)
            frr, far_skilled, far_random = __classification_metrics(prediction, number_of_genuine, number_of_skilled, number_of_random)
            eer, diff_threshold, best_eer = equal_error_rate_with_verification(far_skilled, frr, diff)
            if diff_threshold < diff:
                diff = diff_threshold
                frr_user, far_skilled_user, far_random_user, eer_user = frr, far_skilled, far_random, eer
            if best_eer:
                frr_user, far_skilled_user, far_random_user, eer_user = frr, far_skilled, far_random, eer
                break
        prediction = __prediction_list(global_threshold, list_of_thresholds)
        print(prediction)
        frr_global, far_skilled_global, far_random_global = __classification_metrics(prediction, number_of_genuine, number_of_skilled, number_of_random)
        eer_global = equal_error_rate(far_skilled_global, frr_global)
        results[0].append(frr_user)
        results[1].append(far_skilled_user)
        results[2].append(far_random_user)
        results[3].append(eer_user)
        results[4].append(frr_global)
        results[5].append(far_skilled_global)
        results[6].append(far_random_global)
        results[7].append(eer_global)
        results[8].append(auc_metric)
        results[9].append(threshold)
    return results

def __prediction_list(threshold, prediction_probability):
    prediction = []
    for pred in prediction_probability:
        if pred <= threshold:
            prediction.append(0)
        else:
            prediction.append(1)
    return prediction

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