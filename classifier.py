from sklearn import neighbors
from sklearn import tree as treeClassifier
from sklearn import svm as svmClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from metrics import false_rejection_rate, false_acceptance_rate, equal_error_rate_with_verification, equal_error_rate
import numpy as np
import sklearn.metrics as sk_metrics
from typing import List, Tuple, Dict

def knn(k = 3, weight='distance'):
    return neighbors.KNeighborsClassifier(k, weights=weight)



def tree(data_train, train_classes):
    clf = treeClassifier.DecisionTreeClassifier()
    return clf.fit(data_train, train_classes)
    

def svm(gamma='auto', weights = None, kernel="linear"):
    return svmClassifier.SVC(probability=True, class_weight = weights, kernel = kernel, gamma = gamma)
    

def mlp(solver='adam', alpha=0.0001, hidden_layer_sizes=(100, 8), random_state=1, max_iter=1000):
    return MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter, learning_rate="adaptive")
    

def test(clf, test_sets, test_classes, number_of_genuine, number_of_skilled, number_of_random, global_threshold=0.5):
    scores = {
        "genuine": [],
        "skilled": [],
        "random": []
    }    
    for test_set in test_sets:
        prediction_probability = clf.predict_proba(test_set)
        all_genuine_scores = prediction_probability[:,1]
        all_forgery_scores = prediction_probability[:,0]
        genuine_scores = all_genuine_scores[:number_of_genuine]
        print("genuine_scores")
        print(genuine_scores)
        forgery_scores = all_genuine_scores[number_of_genuine:]
        skilled_scores = forgery_scores[:number_of_skilled]
        print("skilled_scores")
        print(skilled_scores)
        random_scores = forgery_scores[number_of_skilled:]
        print("random_scores")
        print(random_scores)
        scores["genuine"].append(genuine_scores)
        scores["skilled"].append(skilled_scores)
        scores["random"].append(random_scores)
    results = [[], [], [], [], [], [], []]
    result = compute_metrics(scores["genuine"], scores["random"], scores["skilled"], global_threshold)
    print(result)
    results[0].append(result['FRR'])
    results[1].append(result['FAR_skilled'])
    results[2].append(result['FAR_random'])
    results[3].append(result['EER'])
    results[4].append(result['EER_userthresholds'])
    results[5] += (result['auc_list'])
    results[6].append(result['global_threshold'])
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


def compute_metrics(genuine_preds, random_preds, skilled_preds, global_threshold):
    """ Compute metrics given the predictions (scores) of genuine signatures,
    random forgeries and skilled forgeries.

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    random_preds: list of np.ndarray
        A list of predictions of random forgeries (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)
    global_threshold: float
        The global threshold used to compute false acceptance and false rejection rates

    Returns
    -------
    dict
        A dictionary containing:
        'FRR': false rejection rate
        'FAR_random': false acceptance rate for random forgeries
        'FAR_skilled': false acceptance rate for skilled forgeries
        'mean_AUC': mean Area Under the Curve (average of AUC for each user)
        'EER': Equal Error Rate using a global threshold
        'EER_userthresholds': Equal Error Rate using user-specific thresholds
        'auc_list': the list of AUCs (one per user)
        'global_threshold': the optimum global threshold (used in EER)
    """
    all_genuine_preds = np.concatenate(genuine_preds)
    all_random_preds = np.concatenate(random_preds)
    all_skilled_preds = np.concatenate(skilled_preds)

    FRR = 1 - np.mean(all_genuine_preds >= global_threshold)
    FAR_random = 1 - np.mean(all_random_preds < global_threshold)
    FAR_skilled = 1 - np.mean(all_skilled_preds < global_threshold)

    aucs, meanAUC = compute_AUCs(genuine_preds, skilled_preds)

    EER, global_threshold = compute_EER(all_genuine_preds, all_skilled_preds)
    EER_userthresholds = calculate_EER_user_thresholds(genuine_preds, skilled_preds)

    all_metrics = {'FRR': FRR,
                   'FAR_random': FAR_random,
                   'FAR_skilled': FAR_skilled,
                   'mean_AUC': meanAUC,
                   'EER': EER,
                   'EER_userthresholds': EER_userthresholds,
                   'auc_list': aucs,
                   'global_threshold': global_threshold}

    return all_metrics


def compute_AUCs(genuine_preds, skilled_preds):
    """ Compute the area under the curve for the classifiers

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)

    Returns
    -------
    list
        The list of AUCs (one per user)
    float
        The mean AUC

    """
    aucs = []
    for thisRealPreds, thisSkilledPreds in zip(genuine_preds, skilled_preds):
        y_true = np.ones(len(thisRealPreds) + len(thisSkilledPreds))
        y_true[len(thisRealPreds):] = -1
        y_scores = np.concatenate([thisRealPreds, thisSkilledPreds])
        aucs.append(sk_metrics.roc_auc_score(y_true, y_scores))
    meanAUC = np.mean(aucs)
    return aucs, meanAUC.item()


def compute_EER(all_genuine_preds, all_skilled_preds):
    """ Calculate Equal Error Rate with a global decision threshold.

    Parameters
    ----------
    all_genuine_preds: np.ndarray
        Scores for genuine predictions of all users
    all_skilled_preds: np.ndarray
    Scores for skilled forgery predictions of all users

    Returns
    -------
    float:
        The Equal Error Rate
    float:
        The optimum global threshold (a posteriori)

    """

    all_preds = np.concatenate([all_genuine_preds, all_skilled_preds])
    all_ys = np.concatenate([np.ones_like(all_genuine_preds), np.ones_like(all_skilled_preds) * -1])
    fpr, tpr, thresholds = sk_metrics.roc_curve(all_ys, all_preds)

    # Select the threshold closest to (FPR = 1 - TPR)
    t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]
    genuineErrors = 1 - np.mean(all_genuine_preds >= t).item()
    skilledErrors = 1 - np.mean(all_skilled_preds < t).item()
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER, t


def calculate_EER_user_thresholds(genuine_preds, skilled_preds):
    """ Calculate Equal Error Rate with a decision threshold specific for each user

    Parameters
    ----------
    genuine_preds: list of np.ndarray
        A list of predictions of genuine signatures (each element on the list is the
        prediction for one user)
    skilled_preds: list of np.ndarray
        A list of predictions of skilled forgeries (each element on the list is the
        prediction for one user)

    Returns
    -------
    float
        The Equal Error Rate when using user-specific thresholds

    """
    all_genuine_errors = []
    all_skilled_errors = []

    nRealPreds = 0
    nSkilledPreds = 0

    for this_real_preds, this_skilled_preds in zip(genuine_preds, skilled_preds):
        # Calculate user AUC
        y_true = np.ones(len(this_real_preds) + len(this_skilled_preds))
        y_true[len(this_real_preds):] = -1
        y_scores = np.concatenate([this_real_preds, this_skilled_preds])

        # Calculate user threshold
        fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, y_scores)
        # Select the threshold closest to (FPR = 1 - TPR).
        t = thresholds[sorted(enumerate(abs(fpr - (1 - tpr))), key=lambda x: x[1])[0][0]]

        genuineErrors = np.sum(this_real_preds < t)
        skilledErrors = np.sum(this_skilled_preds >= t)

        all_genuine_errors.append(genuineErrors)
        all_skilled_errors.append(skilledErrors)

        nRealPreds += len(this_real_preds)
        nSkilledPreds += len(this_skilled_preds)

    genuineErrors = float(np.sum(all_genuine_errors)) / nRealPreds
    skilledErrors = float(np.sum(all_skilled_errors)) / nSkilledPreds

    # Errors should be nearly equal, up to a small rounding error since we have few examples per user.
    EER = (genuineErrors + skilledErrors) / 2.0
    return EER