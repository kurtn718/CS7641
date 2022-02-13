import plotter
import experiment_reports
import DataLoader as loader
from sklearn import neighbors
from sklearn import metrics
import sklearn.model_selection as model_selection


def titanic_Experiments(grid_search=True):
    X, Y = loader.get_titantic_data()
    x_train, x_test, y_train, y_test = loader.split_data(X,Y,test_size=0.3)

    k_range = range(1,50)
    scores = {}
    score_list = []

    for k in k_range:
        learner = neighbors.KNeighborsClassifier(n_neighbors=k)
        learner.fit(x_train,y_train.ravel())
        y_pred = learner.predict(x_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        score_list.append(metrics.accuracy_score(y_test,y_pred))

    plot = plotter.plot_curve('kNN as k is varied - Titanic',k_range,score_list,x_label='k')
    plot.savefig('kNN-Titanic-kVaried.png', format='png', dpi=150)

    if grid_search:
        model = neighbors.KNeighborsClassifier()
        print(model.get_params().keys())

        k_sizes = range(1,50)
        search_params = {'n_neighbors': k_sizes,
                         "weights": ["uniform", "distance"],
                         'p' : [1,2],
                         'algorithm': ['ball_tree', 'kd_tree', 'brute']}
#        search_params = {'n_neighbors': k_sizes}
        grid_search = model_selection.GridSearchCV(model, param_grid=search_params, refit=True, cv=2,verbose=10,
                                                   scoring=metrics.make_scorer(metrics.accuracy_score))

        grid_search.fit(x_train, y_train)

        best_estimator = grid_search.best_estimator_.fit(x_train, y_train)
        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')

    else:
        best_params = {'algorithm': 'ball_tree', "n_neighbors": 20, 'p': 1, 'weights' : 'distance' }
        model = neighbors.KNeighborsClassifier(**best_params)
        experiment_reports.plotExperiment(model, (X, Y), 'kNN', 'Titanic')

        model = model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def performWineExperiment(grid_search=True):
    X, Y = loader.get_wine_data()
    x_train, X_test, y_train, y_test = loader.split_data(X, Y, test_size=.3, scale_data=True)

    if grid_search:
        model = neighbors.KNeighborsClassifier()
        print(model.get_params().keys())

        k_sizes = range(1,50)
        search_params = {'n_neighbors': k_sizes,
                         "weights": ["uniform", "distance"],
                         'p' : [1,2],
                         'algorithm': ['ball_tree', 'kd_tree', 'brute']}
        grid_search = model_selection.GridSearchCV(model, param_grid=search_params, refit=True, cv=2,verbose=10,
                                                   scoring=metrics.make_scorer(metrics.accuracy_score))

        grid_search.fit(x_train, y_train)

        best_estimator = grid_search.best_estimator_.fit(x_train, y_train)
        best_params = best_estimator.get_params()

        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')
    else:
        best_params = {'algorithm': 'ball_tree', "n_neighbors": 5, "p": 1, "weights" : "uniform" }
        model = neighbors.KNeighborsClassifier(**best_params)
        experiment_reports.plotExperiment(model, (X, Y), 'kNN', 'Wine',scale_data=True)

        model = model.fit(x_train, y_train)
        y_pred = model.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    titanic_Experiments(grid_search=False)
    performWineExperiment(grid_search=False)