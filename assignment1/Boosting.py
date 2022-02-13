import DataLoader as loader
from sklearn import ensemble
from sklearn import metrics
from numpy import arange
import experiment_reports
import sklearn.model_selection as model_selection

def performTitanicExperiment(grid_search=True):
    X, Y = loader.get_titantic_data()
    X_train, X_test, y_train, y_test = loader.split_data(X,Y,test_size=.3)

    if grid_search:
        search_params = {
            "n_estimators": [1, 5, 10, 20, 40, 50, 100, 200],
            "learning_rate": arange(0.1, 2.1, 0.1),
            "algorithm": ["SAMME","SAMME.R"]
        }
        model = ensemble.AdaBoostClassifier(random_state=42)
        grid_search = model_selection.GridSearchCV(model,search_params,cv=5,scoring='accuracy')
        grid_search.fit(X_train,y_train)

        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')
    else:
        best_params = {'algorithm': 'SAMME', 'learning_rate': 1.1, 'n_estimators': 50}
        model = ensemble.AdaBoostClassifier(random_state=42,**best_params)
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        experiment_reports.plotExperiment(model, (X, Y), 'Boosting', 'Titanic', test_size=0.3)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred))
        print("Recall:", metrics.recall_score(y_test, y_pred))

def performWineExperiment(grid_search=True):
    X,Y = loader.get_wine_data()
    X_train, X_test, y_train, y_test = loader.split_data(X,Y,test_size=.3,scale_data=True)

    if grid_search:
        search_params = {
            "n_estimators": [1, 5, 10, 20, 40, 50, 100, 200],
            "learning_rate": arange(0.1, 2.1, 0.1),
            "algorithm": ["SAMME","SAMME.R"]
        }
        model = ensemble.AdaBoostClassifier(random_state=42)
        grid_search = model_selection.GridSearchCV(model,search_params,cv=5,scoring='accuracy')
        grid_search.fit(X_train,y_train)

        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')
    else:
        best_params = {'algorithm': 'SAMME.R', 'learning_rate': 0.9, 'n_estimators': 50}
        model = ensemble.AdaBoostClassifier(random_state=42,**best_params)
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        experiment_reports.plotExperiment(model, (X, Y), 'Boosting', 'Wine', test_size=0.3)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred,average=None))
        print("Recall:", metrics.recall_score(y_test, y_pred,average=None))

if __name__ == '__main__':
    performTitanicExperiment(grid_search=False)
    performWineExperiment(grid_search=False)

