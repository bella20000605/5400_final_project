from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(results, metric):

    y_true = [x[2] for x in results] 
    y_pred = [x[0] == x[1] for x in results]

    if metric == 'precision':
        return precision_score(y_true, y_pred)
    elif metric == 'recall':
        return recall_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred)
    elif metric == 'confusion_matrix':
        return confusion_matrix(y_true, y_pred, labels=["scar", "notscar"])
    else:
        raise ValueError("Unsupported metric. Choose 'precision', 'recall'.'confusion_matrix', or 'f1'.")

