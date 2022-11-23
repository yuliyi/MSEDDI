import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, \
    precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize


def evaluate(y_pred, y_test, pred_score, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type,), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    result_all[0] = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    result_all[1] = roc_aupr_score(y_test, pred_score, average='micro')
    result_all[2] = roc_auc_score(y_test, pred_score, average='micro')
    result_all[3] = f1_score(y_test, y_pred, average='macro')
    result_all[4] = precision_score(y_test, y_pred, average='macro')
    result_all[5] = recall_score(y_test, y_pred, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_test.take([i], axis=1).ravel(), y_pred.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_test.take([i], axis=1).ravel(), y_pred.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_test.take([i], axis=1).ravel(), y_pred.take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_test.take([i], axis=1).ravel(), y_pred.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_test.take([i], axis=1).ravel(), y_pred.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_test.take([i], axis=1).ravel(), y_pred.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


def evaluate_valid(y_pred, y_test, pred_score, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type,), dtype=float)
    result_all[0] = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    result_all[1] = roc_aupr_score(y_test, pred_score, average='micro')
    result_all[2] = roc_auc_score(y_test, pred_score, average='micro')
    result_all[3] = f1_score(y_test, y_pred, average='macro')
    result_all[4] = precision_score(y_test, y_pred, average='macro')
    result_all[5] = recall_score(y_test, y_pred, average='macro')
    return result_all


def evaluate_train(y_pred, y_test, pred_score, event_num):
    all_eval_type = 3
    result_all = np.zeros((all_eval_type,), dtype=float)
    result_all[0] = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    result_all[1] = roc_aupr_score(y_test, pred_score, average='micro')
    result_all[2] = roc_auc_score(y_test, y_pred, average='micro')
    return result_all


def roc_aupr_score(y_true, y_score, average="macro"):

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average is None:
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
            return binary_metric(y_true, y_score)
        if average == "macro":
            n_classes = y_score.shape[1]
            score = np.zeros(n_classes)
            for c in range(n_classes):
                y_true_c = y_true[c]
                y_score_c = y_score[c]
                score[c] = binary_metric(y_true_c, y_score_c)
            return np.average(score)

    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
