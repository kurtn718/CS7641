import DataLoader as loader
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import experiment_reports
import sklearn.model_selection as model_selection

# Uncomment if we want to visualize tree
#from six import StringIO
#from IPython.display import Image
#from sklearn.tree import export_graphviz
#import pydotplus

def performTitanicExperiment(grid_search=True):
    X, Y = loader.get_titantic_data()
    X_train, X_test, y_train, y_test = loader.split_data(X,Y,test_size=.3)

    if grid_search:
        search_params = {
            "max_depth": [3, 5, 7, 9, 11, 13],
            "min_samples_split": [1,3,5,7,9,11,13,15,17,19,21],
            "criterion": ["gini","entropy"]
        }
        model = DecisionTreeClassifier(random_state=42)
        grid_search = model_selection.GridSearchCV(model,search_params,cv=5,scoring='accuracy')
        grid_search.fit(X_train,y_train)

        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')
    else:
        model = DecisionTreeClassifier(random_state=42,criterion='entropy',max_depth=5,min_samples_split=7)
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        experiment_reports.plotExperiment(model, (X, Y), 'Decision Tree', 'Titanic', test_size=0.3)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred))
        print("Recall:", metrics.recall_score(y_test, y_pred))

def performWineExperiment(grid_search=True):
    X,Y = loader.get_wine_data()
    X_train, X_test, y_train, y_test = loader.split_data(X,Y,test_size=.3,scale_data=True)

    if grid_search:
        search_params = {
            "max_depth": [3, 5, 7, 9, 11, 13],
            "min_samples_split": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            "criterion": ["gini", "entropy"]
        }
        model = DecisionTreeClassifier(random_state=42)
        grid_search = model_selection.GridSearchCV(model, search_params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        print(f'Best parameters {grid_search.best_params_}')
        print(f'Accuracy score of best estimator {grid_search.best_score_}')
    else:
        model = DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=5, min_samples_split=3)
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        experiment_reports.plotExperiment(model, (X, Y), 'Decision Tree', 'Wine', test_size=0.3)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred,average=None))
        print("Recall:", metrics.recall_score(y_test, y_pred,average=None))


def plotDecisionTree(model,feature_columns,class_names,output_file):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_columns,class_names=['0','1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(output_file)
    Image(graph.create_png())


if __name__ == '__main__':
    performTitanicExperiment(grid_search=False)
    performWineExperiment(grid_search=False)
