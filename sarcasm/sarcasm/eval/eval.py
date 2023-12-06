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
    else:
        raise ValueError("Unsupported metric. Choose 'precision', 'recall', or 'f1'.")

