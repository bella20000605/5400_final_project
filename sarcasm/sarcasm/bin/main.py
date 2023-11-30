import argparse
import pandas as pd
import numpy as np
import statistics
from sarcasm.sarcasm_model import Logistic
from sarcasm.sarcasm_model import NaiveBayes
from sarcasm.sarcasm_model import LSTMDetector
from sarcasm.sarcasm_model import RandomForestSarcasm
from sarcasm.sarcasm_translation import sentiment_score_check
from sarcasm.sarcasm_translation import sentiment_score_adjective
from sarcasm.sarcasm_translation import antonyms_for
from sarcasm.sarcasm_translation import adjective_translation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="code to run the sarcasm detection and sarcasm translation model"
    )
    parser.add_argument("-t", "--text", required=True, help="text string to predict")

    #Model has to be either "LG", "NB", "LTSM", "RF" or "All"
    parser.add_argument("-m", "--model", required=True, help="model to choose")
    parser.add_argument("-f", "--flag", required=True, help="flag whether to translate text")

    args = parser.parse_args()

    data = {'text': [args.text]}
    x_test = pd.DataFrame(data)
    x_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    x_train_1 = np.ravel(x_train)
    y_train_1 = np.ravel(y_train)
    x_test_1 = np.ravel(x_test)

    results = []

    if args.model == "LTSM" or args.model == "All":
        sarcasm_mod_LTSM = LSTMDetector()
        model, tokenizer = sarcasm_mod_LTSM.train(x_train,y_train)
        predicition_LTSM = sarcasm_mod_LTSM.test(x_test,model,tokenizer)
        results.append(predicition_LTSM[0])

    sarcasm_mod_LG = Logistic()
    tfidf, LogisticModel = sarcasm_mod_LG.train(x_train_1,y_train_1)
    prediction_LG = sarcasm_mod_LG.test(LogisticModel,tfidf,x_test_1)
    results.append(prediction_LG[0])

    sarcasm_mod_NB = NaiveBayes()
    vectorizer,classifier = sarcasm_mod_NB.train(x_train_1,y_train_1)
    prediction_NB = sarcasm_mod_NB.test(classifier,vectorizer,x_test_1)
    results.append(prediction_NB[0])

    sarcasm_mod_RF = RandomForestSarcasm()
    tfidf_RF, rf_model = sarcasm_mod_RF.train(x_train_1,y_train_1)
    prediction_RF = sarcasm_mod_RF.test(rf_model,tfidf_RF,x_test_1)
    results.append(prediction_RF[0])


    if args.model == "LG":
        final_result = prediction_LG[0]
        print(f"Logistic model predicts the text to be {prediction_LG[0]}")

    elif args.model == "NB":
        final_result = prediction_NB[0]
        print(f"Naive Bayes model predicts the text to be {prediction_NB[0]}")

    elif args.model == "RF":
        final_result = prediction_RF[0]
        print(f"Random Forest model predicts the text to be {prediction_RF[0]}")
    
    elif args.model == "LTSM":
        final_result = predicition_LTSM[0]
        print(f"LTSM model predicts the text to be {predicition_LTSM[0]}")
    
    elif args.model == "All":
        final_result = statistics.mode(results)
        print(f"Based all models results, the text is predicted to be {final_result}")
    
    else:
        raise ValueError("-n needs to be LG, NB, RF, LTSM or All")
    
    if args.flag == "T":
        if final_result == 'sarc':
            if sentiment_score_check(args.text):
                adj_list = sentiment_score_adjective(args.text)
                non_sarcasm_candidates = adjective_translation(args.text, adj_list)
                if non_sarcasm_candidates != None:
                    print("Here is the potential true meanings of the text\n")
                    print(non_sarcasm_candidates)
        else:
            print('Process ends.')
    else:
        print('No translation needs flagged. Process ends.')
    

    

    


