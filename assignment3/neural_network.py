import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neural_network import MLPClassifier
import DataLoader as loader
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.mixture import GaussianMixture as EM
from sklearn.cluster import KMeans

def compare_fit_time(n ,full_fit ,pca_fit ,ica_fit ,rca_fit ,rfc_fit ,title):
    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Training Time (s)")
    plt.plot(n, full_fit, '-', color="k", label="Full Dataset")
    plt.plot(n, pca_fit, '-', color="b", label="PCA")
    plt.plot(n, ica_fit, '-', color="r", label="ICA")
    plt.plot(n, rca_fit, '-', color="g", label="RCA")
    plt.plot(n, rfc_fit, '-', color="m", label="RFC")
    plt.legend(loc="best")
    plt.show()

def compare_pred_time(n ,full_pred, pca_pred, ica_pred, rca_pred, rfc_pred, title):
    plt.figure()
    plt.title("Model Prediction Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Prediction Time (s)")
    plt.plot(n, full_pred, '-', color="k", label="Full Dataset")
    plt.plot(n, pca_pred, '-', color="b", label="PCA")
    plt.plot(n, ica_pred, '-', color="r", label="ICA")
    plt.plot(n, rca_pred, '-', color="g", label="RCA")
    plt.plot(n, rfc_pred, '-', color="m", label="RFC")
    plt.legend(loc="best")
    plt.show()


def compare_learn_time(n ,full_learn, pca_learn, ica_learn, rca_learn, rfc_learn, title):
    plt.figure()
    plt.title("Model Learning Rates: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.plot(n, full_learn, '-', color="k", label="Full Dataset")
    plt.plot(n, pca_learn, '-', color="b", label="PCA")
    plt.plot(n, ica_learn, '-', color="r", label="ICA")
    plt.plot(n, rca_learn, '-', color="g", label="RCA")
    plt.plot(n, rfc_learn, '-', color="m", label="RFC")
    plt.legend(loc="best")
    plt.show()


def final_classifier_evaluation(clf, X_train, X_test, y_train, y_test):
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    start_time = timeit.default_timer()
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time

    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.5f}".format(training_time))
    print("Model Prediction Time (s): " + "{:.5f}\n".format(pred_time))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
#    plt.figure()
#    plt.show()


def plot_learning_curve(clf, X, y, title="Insert Title"):
    n = len(y)
    train_mean = [];
    train_std = []  # model performance score (f1)
    cv_mean = [];
    cv_std = []  # model performance score (f1)
    fit_mean = [];
    fit_std = []  # model fit/training time
    pred_mean = [];
    pred_std = []  # model test/prediction times
    train_sizes = (np.linspace(.05, 1.0, 20) * n).astype('int')

    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X[idx, :]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='accuracy', n_jobs=-1, return_train_score=True)

        train_mean.append(np.mean(scores['train_score']));
        train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score']));
        cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time']));
        fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time']));
        pred_std.append(np.std(scores['score_time']))

    train_mean = np.array(train_mean);
    train_std = np.array(train_std)
    cv_mean = np.array(cv_mean);
    cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean);
    fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean);
    pred_std = np.array(pred_std)

    plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)

    return train_sizes, train_mean, fit_mean, pred_mean


def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    plt.figure()
    plt.title("Learning Curve: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Accuracy Score")
    plt.fill_between(train_sizes, train_mean - 2 * train_std, train_mean + 2 * train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2 * cv_std, cv_mean + 2 * cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()


def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    plt.figure()
    plt.title("Modeling Time: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2 * fit_std, fit_mean + 2 * fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()


def addclusters(X, km_lables, em_lables):
    df = pd.DataFrame(X)
    df['KM Cluster'] = km_lables
    df['EM Cluster'] = em_lables
    col_1hot = ['KM Cluster', 'EM Cluster']
    df_1hot = df[col_1hot]
    df_1hot = pd.get_dummies(df_1hot).astype('category')
    df_others = df.drop(col_1hot, axis=1)
    df = pd.concat([df_others, df_1hot], axis=1)
    new_X = np.array(df.values, dtype='int64')

    return new_X

def run_RFC(X,y,df_original):
    rfc = RFC(n_estimators=500,min_samples_leaf=round(len(X)*.01),random_state=42,n_jobs=-1)
    imp = rfc.fit(X,y).feature_importances_
    print(imp)
    imp = pd.DataFrame(imp,columns=['Feature Importance'],index=df_original.columns) #.columns[2::])
    imp.sort_values(by=['Feature Importance'],inplace=True,ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    imp = imp[imp['Cum Sum']<=0.95]
    top_cols = imp.index.tolist()
    return imp, top_cols

def performNNExperimentWithClusters():
    km = KMeans(n_clusters=5,random_state=42).fit(titanicX)
    km_labels = km.labels_
    em = EM(n_components=5, covariance_type='diag', n_init=1, warm_start=True, random_state=100).fit(titanicX)
    em_labels = em.predict(titanicX)

    clust_full = addclusters(titanicX, km_labels, em_labels)
    clust_pca = addclusters(pca_titanic, km_labels, em_labels)
    clust_ica = addclusters(ica_titanic, km_labels, em_labels)
    clust_rca = addclusters(rca_titanic, km_labels, em_labels)
    clust_rfc = addclusters(rfc_titanic, km_labels, em_labels)

    # Original, full dataset
    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_full), np.array(titanicY), test_size=0.20)
    full_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',early_stopping=False,max_iter=4000,random_state=42)

    train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(full_est, X_train,
                                                                                                    y_train,
                                                                                                    title="Neural Net Titanic with Clusters: Full")
    final_classifier_evaluation(full_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_pca), np.array(titanicY), test_size=0.20)
    pca_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', early_stopping=False, max_iter=4000, random_state=42)
    train_samp_pca, NN_train_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Titanic with Clusters: PCA")
    final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_ica), np.array(titanicY), test_size=0.20)
    ica_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', early_stopping=False, max_iter=4000, random_state=42)
    train_samp_ica, NN_train_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est, X_train,
                                                                                                y_train,
                                                                                                title="Neural Net Titanic with Clusters: ICA")
    final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_rca), np.array(titanicY), test_size=0.20)
    rca_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', early_stopping=False, max_iter=4000, random_state=42)
    train_samp_rca, NN_train_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est, X_train, y_train,
                                                                                                title="Neural Net Titanic with Clusters: RCA")
    final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(clust_rfc), np.array(titanicY), test_size=0.20)
    rfc_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', early_stopping=False, max_iter=4000, random_state=42)
    train_samp_rfc, NN_train_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est, X_train, y_train,
                                                                                                title="Neural Net Titanic with Clusters: RFC")
    final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test)

    compare_fit_time(train_samp_full, NN_fit_time_full, NN_fit_time_pca, NN_fit_time_ica,
                     NN_fit_time_rca, NN_fit_time_rfc, 'Titanic Dataset')
    compare_pred_time(train_samp_full, NN_pred_time_full, NN_pred_time_pca, NN_pred_time_ica,
                      NN_pred_time_rca, NN_pred_time_rfc, 'Titanic Dataset')
    compare_learn_time(train_samp_full, NN_train_score_full, NN_train_score_pca, NN_train_score_ica,
                       NN_train_score_rca, NN_train_score_rfc, 'Titanic Dataset')

titanicX, titanicY = loader.get_titantic_data()
titanicY = titanicY.ravel()
pca_titanic = PCA(n_components=7, random_state=42).fit_transform(titanicX)
ica_titanic = ICA(n_components=5, random_state=42).fit_transform(titanicX)
rca_titanic = RCA(n_components=5, random_state=42).fit_transform(titanicX)
print(titanicX.shape)
print(titanicY.shape)
x_df = pd.DataFrame(data=titanicX,
                          columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title"])
y_df = pd.DataFrame(data=titanicY, columns=["Survived"])
df_titanic = pd.concat([x_df, y_df], axis=1)
imp_titanic, topcols_titanic = run_RFC(titanicX,titanicY,x_df)
rfc_titanic = df_titanic[topcols_titanic]
rfc_titanic = np.array(rfc_titanic.values,dtype='int64')

def performNNExperiment():
    X_train, X_test, y_train, y_test = train_test_split(np.array(titanicX), np.array(titanicY), test_size=0.30)
    full_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',early_stopping=False,max_iter=4000,random_state=42)
    train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(full_est, X_train, y_train,
                                                                                                    title="Titanic Neural Network")
    final_classifier_evaluation(full_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(pca_titanic), np.array(titanicY), test_size=0.30)
    pca_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',early_stopping=False,max_iter=4000,random_state=42)
    train_samp_pca, NN_train_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est, X_train,
                                                                                                y_train
                                                                                                ,
                                                                                                title="Neural Net Titanic: PCA")
    final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(ica_titanic), np.array(titanicY), test_size=0.30)
    ica_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',early_stopping=False,max_iter=4000,random_state=42)

    train_samp_ica, NN_train_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est, X_train,
                                                                                                y_train
                                                                                                ,
                                                                                                title="Neural Net Titanic: ICA")
    final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(rca_titanic), np.array(titanicY), test_size=0.30)
    rca_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',early_stopping=False,max_iter=4000,random_state=42)
    train_samp_rca, NN_train_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est, X_train,
                                                                                                y_train
                                                                                                ,
                                                                                                title="Neural Net Titanic: RCA")
    final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test)

    X_train, X_test, y_train, y_test = train_test_split(np.array(rfc_titanic), np.array(titanicY), test_size=0.30)
    rfc_est = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',early_stopping=False,max_iter=4000,random_state=42)
    train_samp_rfc, NN_train_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est, X_train,
                                                                                                y_train
                                                                                                ,
                                                                                                title="Neural Net Titanic: RFC")
    final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test)

    compare_fit_time(train_samp_full, NN_fit_time_full, NN_fit_time_pca, NN_fit_time_ica,
                     NN_fit_time_rca, NN_fit_time_rfc, 'Titanic Dataset')
    compare_pred_time(train_samp_full, NN_pred_time_full, NN_pred_time_pca, NN_pred_time_ica,
                      NN_pred_time_rca, NN_pred_time_rfc, 'Titanic Dataset')
    compare_learn_time(train_samp_full, NN_train_score_full, NN_train_score_pca, NN_train_score_ica,
                       NN_train_score_rca, NN_train_score_rfc, 'Titanic Dataset')

if __name__ == '__main__':
#    performNNExperiment()
    performNNExperimentWithClusters()