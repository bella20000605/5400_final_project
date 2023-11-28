from sklearn.metrics import precision_score, recall_score, f1_score
from sarcasm.sarcasm_model import Logistic
from sarcasm.sarcasm_model import NaiveBayes


def Logistic_Eval(X_train, y_train, X_test, y_test,metric):
    """
    Evaluate the results for Logistic Regression model.
    """
    sarcasm_model = Logistic()
    tfidf, LogisticModel = sarcasm_model.train(X_train,y_train)
    y_pred = sarcasm_model.test(LogisticModel,tfidf,X_test)

    if metric == 'precision':
        return precision_score(y_test, y_pred, average='binary')
    elif metric == 'recall':
        return recall_score(y_test, y_pred, average='binary')
    elif metric == 'f1':
        return f1_score(y_test, y_pred, average='binary')
    else:
        raise ValueError("Unknown metric: {}".format(metric))
    
def NaiveBayes_Eval(X_train, y_train, X_test, y_test,metric):
    """
    Evaluate the results for Naive Bayes model.
    """
    sarcasm_model = NaiveBayes(var_smoothing=0.1)
    vectorizer,classifier= sarcasm_model.train(X_train,y_train)
    y_pred = sarcasm_model.test(classifier,vectorizer,X_test)

    if metric == 'precision':
        return precision_score(y_test, y_pred, average='binary')
    elif metric == 'recall':
        return recall_score(y_test, y_pred, average='binary')
    elif metric == 'f1':
        return f1_score(y_test, y_pred, average='binary')
    else:
        raise ValueError("Unknown metric: {}".format(metric))
