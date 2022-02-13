import DataLoader as loader
import sklearn.model_selection as model_selection
from sklearn import metrics
from sklearn import svm
import experiment_reports


def performTitanicExperiment(grid_search=True):
    X, Y = loader.get_titantic_data()
    X_train, X_test, y_train, y_test = loader.split_data(X,Y)

    if grid_search:
        search_params = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.001, 0.01, 0.1, 1.],
            "gamma": ["scale", "auto"],
        }
        model = svm.SVC(random_state=42,class_weight="balanced",probability=True)

        grid_search = model_selection.GridSearchCV(model,
                                                   search_params,
                                                   cv=5,
                                                   scoring='accuracy',
                                                   verbose=10)

        grid_search.fit(X_train, y_train.ravel())

        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')
    else:
        best_params = {'kernel': 'poly', 'C': 0.1, 'gamma': 'scale'}
        model = svm.SVC(**best_params)
        experiment_reports.plotExperiment(model, (X, Y), 'SVM', 'Titanic',test_size=0.3)
        y_pred = model.predict(X_train)
        print("Accuracy:", metrics.accuracy_score(y_train, y_pred))
        print("Precision:", metrics.precision_score(y_train, y_pred))
        print("Recall:", metrics.recall_score(y_train, y_pred))

    #Create a svm Classifier
#    model = svm.SVC(kernel='rbf',gamma=0.0001,C=10000,degree=3) # Linear Kernel  #rbf other kernal

    #clf = svm.SVC(kernel='linear',C=1,degree=5) # Linear Kernel  #rbf other kernal

    #Train the model using the training sets
 #   model.fit(X_train, y_train)

    #Predict the response for test dataset
#    y_pred = model.predict(X_test)
#    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#    print("Precision:",metrics.precision_score(y_test, y_pred))
#    print("Recall:",metrics.recall_score(y_test, y_pred))

def performWineExperiment(grid_search=True):
    X, Y = loader.get_titantic_data()
    x_train, x_test, y_train, y_test = loader.split_data(X,Y,test_size=0.3,scale_data=True)

    if grid_search:
        search_params = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.001, 0.01, 0.1, 1.],
            "gamma": ["scale", "auto"],
        }
        model = svm.SVC(random_state=42,class_weight="balanced",probability=True)

        grid_search = model_selection.GridSearchCV(model,
                                                   search_params,
                                                   cv=5,
                                                   scoring='accuracy',
                                                   verbose=10)

        grid_search.fit(x_train, y_train.ravel())

        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')
    else:
        best_params = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
        model = svm.SVC(**best_params)
        experiment_reports.plotExperiment(model, (X, Y), 'SVM', 'Wine', test_size=0.3)
        y_pred = model.predict(x_train)
        print("Accuracy:", metrics.accuracy_score(y_train, y_pred))
        print("Precision:", metrics.precision_score(y_train, y_pred))
        print("Recall:", metrics.recall_score(y_train, y_pred))
        pass

if __name__ == '__main__':
    performTitanicExperiment(grid_search=False)
    performWineExperiment(grid_search=False)