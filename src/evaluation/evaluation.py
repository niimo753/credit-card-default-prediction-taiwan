from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

def evaluation(y_true, y_pred, y_pred_prob, scoring=['roc_auc', 'accuracy', 'f1', 'precision','recall']):
    scores = {'accuracy': accuracy_score,
              'f1': f1_score,
              'recall': recall_score,
              'precision': precision_score,
              'roc_auc': roc_auc_score}
    
    result = {}
    for method in scoring:
        if method == 'roc_auc':
            result[method] = scores[method](y_true, y_pred_prob.T[1])
        else:
            result[method] = scores[method](y_true, y_pred)

    return result

def cross_validation(X, y, estimator, cv=5, random_state=42, methods=['roc_auc', 'accuracy', 'f1', 'precision','recall'],
                     avg_output=True):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = {method: [] for method in methods}
    for train_index, test_index in skf.split(X, y):
        x_train, y_train = X.iloc[train_index], y.iloc[train_index]
        x_test, y_test = X.iloc[test_index], y.iloc[test_index]

        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)
        y_pred_proba = estimator.predict_proba(x_test)

        score = evaluation(y_true=y_test, y_pred=y_pred, y_pred_prob=y_pred_proba)
        for method in scores.keys():
            scores[method].append(score[method])
    
    if avg_output:
        scores = {key: np.mean(values) for key, values in scores.items()}

    return scores

def modelling_evaluation(estimator, x_train, y_train, x_test, y_test, scoring=['roc_auc', 'accuracy', 'f1', 'precision','recall']):
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    y_pred_proba = estimator.predict_proba(x_test)
    scores = evaluation(y_true=y_test, y_pred=y_pred, y_pred_prob=y_pred_proba, scoring=scoring)
    return scores