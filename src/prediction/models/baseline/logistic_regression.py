import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from src.utils import utils
import random


def run(X_train, X_test, y_train, lr_settings):
    lr_model = LogisticRegression(solver="saga", penalty="l1", class_weight="balanced", max_iter=5000)

    # K-Fold Cross Validation: START #
    # hyper-parameter tuning using K-Fold Cross Validation with K = 5;
    # shuffle the data with given random seed before splitting into batches
    tuning_parameters = {"C": lr_settings["C"]}
    scoring_param = "accuracy"
    print(f"Tuning hyper-parameters {tuning_parameters} based on {scoring_param}")

    # use stratified k-fold to ensure each set contains approximately the same percentage of samples of each target class as the complete set.
    # TODO: should we use StratifiedShuffleSplit instead? What is the difference?
    kfold_cv_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

    # refit=True : retrain the best model on the full training dataset
    cv_model = GridSearchCV(estimator=lr_model, param_grid=tuning_parameters, scoring=scoring_param,
                            cv=kfold_cv_model, verbose=2, refit=True)
    cv_model.fit(X_train, y_train)

    # The best values chosen by KFold-cross-validation
    print("Best parameters in trained model = ", cv_model.best_params_)
    print("Best score in trained model = ", cv_model.best_score_)
    classifier = cv_model.best_estimator_
    # K-Fold Cross Validation: END #

    return classifier.predict_proba(X_test)
