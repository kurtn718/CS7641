import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
from sklearn import neural_network
from sklearn import metrics
import DataLoader as loader
import plotter
import experiment_reports

def titanic_Experiments(grid_search=True):
    X, Y = loader.get_titantic_data()
    x_train, x_test, y_train, y_test = loader.split_data(X,Y,test_size=0.3,scale_data=True)

    if grid_search:
        model = neural_network.MLPClassifier(max_iter=4000,random_state=42,early_stopping=True)
        hidden_layer_sizes = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,50,100,200,300,400]
        activation_functions = ['logistic', 'relu']
        alphas = [.000001,.00001,.0001,.001,.002,.004,.008,.016,.032,0.64,0.128,.256,.512,1,2,4,8,16]
        learning_rates = [.000001,.00001,.0001,.001,.002,.004,.008,.016,.032,0.64,0.128]

#    hidden_layer_sizes = [32,64,128,256]
#    activation_functions = ['logistic', 'relu']
#    alphas = [.001,.002,.004]
#    learning_rates = [.000001,.00001]

        search_params = {'activation': ['relu', 'logistic'],
                  'learning_rate_init': learning_rates,
                  'alpha' : alphas,
                  'hidden_layer_sizes': hidden_layer_sizes}

        grid_search = model_selection.GridSearchCV(model,param_grid=search_params,refit=True,verbose=10,cv=5,scoring=metrics.make_scorer(metrics.accuracy_score))
        print(grid_search.estimator.get_params().keys())

        grid_search.fit(x_train.values, y_train.values.ravel())
        results = pd.DataFrame(grid_search.cv_results_)
        print(results)
        results.to_csv('grid-search-results.csv')

        best_estimator = grid_search.best_estimator_.fit(x_train.values, y_train.values.ravel())
        best_params = pd.DataFrame([best_estimator.get_params()])
        print(best_params)
        best_params.to_csv('best-params.csv')

        print("Activation: " + best_params['activation'])
        print("Learning Rate: " + str(best_params['learning_rate_init']))
        print("Alpha: " + str(best_params['alpha']))
        print("Hidden Layer Sizes: " + str(best_params['hidden_layer_sizes']))
    else:
        best_params = {"activation" : "logistic",
                       "alpha" : 0.016,
                       "learning_rate_init" : 0.32,
                       "hidden_layer_sizes" : (100) }
## Attempted different solver, and hidden_layer_sizes
##                       "hidden_layer_sizes" : (128,64,128),
##                       "solver" : "lbfgs"}
        model = neural_network.MLPClassifier(max_iter=4000,random_state=42,early_stopping=True,verbose=10,**best_params)
        print(model.get_params())
        experiment_reports.plotExperiment(model,(X,Y),'Neural Network','Titanic')

        #model = neural_network.MLPClassifier()
        #model.fit(x_train,y_train)
        y_pred = model.predict(x_train)
        print("Accuracy:", metrics.accuracy_score(y_train, y_pred))
        print("Precision:", metrics.precision_score(y_train, y_pred))
        print("Recall:", metrics.recall_score(y_train, y_pred))

def wine_Experiments(grid_search=True):
    X, Y = loader.get_wine_data()
    x_train, x_test, y_train, y_test = loader.split_data(X,Y,test_size=0.3,scale_data=True)

    if grid_search:
        model = neural_network.MLPClassifier(random_state=42,early_stopping=False)
        max_iter_sizes = [100, 250, 1000, 2000, 4000]
        hidden_layer_sizes = [1,2,4,8,16,32,50,64,100,128]
        activation_functions = ['logistic', 'relu','tanh']
        alphas = [.000001,.00001,.0001,.001,.002,.004,.008,.016,.032,0.64,0.128,.256]

#    hidden_layer_sizes = [32,64,128,256]
#    activation_functions = ['logistic', 'relu']
#    alphas = [.001,.002,.004]
#    learning_rates = [.000001,.00001]

        search_params = {'activation': ['relu', 'logistic','tanh'],
                  'alpha' : alphas,
                  'hidden_layer_sizes': hidden_layer_sizes,
                  'max_iter' : max_iter_sizes }

        grid_search = model_selection.GridSearchCV(model,param_grid=search_params,refit=True,verbose=10,cv=5,scoring=metrics.make_scorer(metrics.accuracy_score))
        print(grid_search.estimator.get_params().keys())

        grid_search.fit(x_train.values, y_train.values.ravel())
        results = pd.DataFrame(grid_search.cv_results_)
        print(results)
        results.to_csv('wine-nn-grid-search-results.csv')

        best_estimator = grid_search.best_estimator_.fit(x_train.values, y_train.values.ravel())
        best_params = pd.DataFrame([best_estimator.get_params()])
        print(best_params)
        best_params.to_csv('wine-nnbest-params.csv')

        print("Activation: " + best_params['activation'])
        print("Alpha: " + str(best_params['alpha']))
        print("Hidden Layer Sizes: " + str(best_params['hidden_layer_sizes']))
    else:
        best_params = {"activation" : "tanh",
                       "alpha" : 0.01,
                       "hidden_layer_sizes" : 50,
                       "max_iter" : 250 }
        model = neural_network.MLPClassifier(random_state=42,early_stopping=False,**best_params)
        print(model.get_params())



        experiment_reports.plotExperiment(model,(X,Y),'Neural Network','Wine')
        #model = neural_network.MLPClassifier()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred,average=None))
        print("Recall:", metrics.recall_score(y_test, y_pred,average=None))

if __name__ == '__main__':
    titanic_Experiments(grid_search=False)
    wine_Experiments(grid_search=False)


#    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#    print("Precision:", metrics.precision_score(y_test, y_pred))
#    print("Recall:", metrics.recall_score(y_test, y_pred))
