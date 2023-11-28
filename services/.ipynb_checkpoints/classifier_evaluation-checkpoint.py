from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)

# Classe para avaliar os classificadores
class ClassifierWrapper:
    def __init__(self, classifier, parameters, random_state=42):
        self.classifier = classifier
        self.parameters = parameters
        self.random_state = random_state
        self.best_params = {}
        self.best_score = 0
        self.best_estimator = None

    def grid_search(self, x_train, y_train, cv=10, n_jobs=-1, verbose=1):
        clf_cv = GridSearchCV(estimator=self.classifier, param_grid=self.parameters, scoring="accuracy", n_jobs=n_jobs, verbose=verbose, cv=cv)
        clf_cv.fit(x_train, y_train)
        
        self.best_params = clf_cv.best_params_
        self.best_score = clf_cv.best_score_
        self.best_estimator = clf_cv.best_estimator_
        
        print("Best Parameters:")
        print(self.best_params)
        print("Best Score:")
        print(self.best_score)
        print("Best Estimator:")
        print(self.best_estimator)

    def train_and_print_scores(self, x_train, y_train, x_test, y_test):
        self.classifier = self.classifier.set_params(**self.best_params)
        self.classifier.fit(x_train, y_train)
        
        self.print_score(x_train, y_train, train=True)
        self.print_score(x_test, y_test, train=False)

    def print_score(self, x, y, train=True):
        pred = self.classifier.predict(x)
        clf_report = pd.DataFrame(classification_report(y, pred, output_dict=True))
        
        if train:
            print("Train Result:\n==========================================================================")
        else:
            print("Test Result:\n==========================================================================")

        print(f"Accuracy Score: {accuracy_score(y, pred) * 100:.2f}%")
        print("__________________________________________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("__________________________________________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y, pred)}\n")

    def avaliar_classificador(classifier, params, x, y, n_splits=10):
        resultados = []
    
        for i in range(5):
            print(i)
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=i)
    
            clf = classifier(**params)
            scores = cross_val_score(clf, x, y, cv=kfold)
            resultados.append(scores.mean())
    
        return resultados