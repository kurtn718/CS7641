import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import  homogeneity_score, silhouette_score

import DataLoader
import DataLoader as loader
import seaborn

np.random.seed(42)

# Code adapted from: https://github.com/kylewest520/CS-7641---Machine-Learning/blob/master/Assignment%203%20Unsupervised%20Learning/CS%207641%20HW3%20Code.ipynb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture as EM

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12


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
        scores = cross_validate(clf, X_subset, y_subset, cv=10, scoring='f1', n_jobs=-1, return_train_score=True)

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
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2 * train_std, train_mean + 2 * train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2 * cv_std, cv_mean + 2 * cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    #plt.show()


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
    #plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


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
    plt.figure()
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    #plt.show()


def cluster_predictions(Y, clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    #    assert max(pred) == max(Y)
    #    assert min(pred) == min(Y)
    return pred


def pairwiseDistCorr(X1, X2):
    assert X1.shape[0] == X2.shape[0]

    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]

def run_kmeans(X, y, title):
    kclusters = list(np.arange(2, 50, 1))
    sil_scores = [];
    f1_scores = [];
    homo_scores = [];
    train_times = []
    inertias = []

    for k in kclusters:
        start_time = timeit.default_timer()
        km = KMeans(n_clusters=k, random_state=42).fit(X)
        end_time = timeit.default_timer()
        inertias.append(km.inertia_)
        train_times.append(end_time - start_time)
        sil_scores.append(silhouette_score(X, km.labels_))
        y_mode_vote = cluster_predictions(y, km.labels_)
        # For titanic we don't need to specify average
#        f1_scores.append(f1_score(y, y_mode_vote))
#        homo_scores.append(homogeneity_score(y, km.labels_))

    # elbow curve for silhouette score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, sil_scores)
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score for KMeans: ' + title)
#    plt.show()
    plt.savefig("kmeans-" + title + "-silhouette.png")

    # Intertia
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, inertias)
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia for KMeans: ' + title)
#    plt.show()
    plt.savefig("inertia-" + title + '-kmeans.png')


    # plot homogeneity scores
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(kclusters, homo_scores)
#    plt.grid(True)
#    plt.xlabel('No. Clusters')
#    plt.ylabel('Homogeneity Score')
#    plt.title('Homogeneity Scores KMeans: ' + title)
#    plt.show()


#     # plot f1 scores
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(kclusters, f1_scores)
#     plt.grid(True)
#     plt.xlabel('No. Clusters')
#     plt.ylabel('F1 Score')
#     plt.title('F1 Scores KMeans: '+ title)
#     plt.show()

#     # plot model training time
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(kclusters, train_times)
#     plt.grid(True)
#     plt.xlabel('No. Clusters')
#     plt.ylabel('Training Time (s)')
#     plt.title('KMeans Training Time: '+ title)
#     plt.show()

def evaluate_kmeans(km, X, y):
    start_time = timeit.default_timer()
    km.fit(X, y)
    end_time = timeit.default_timer()
    training_time = end_time - start_time

    print(km.labels_)
    y_mode_vote = cluster_predictions(y, km.labels_)
    # don't neet multiclass for titanic #average=None
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    print("Model Evaluation Metrics Using Mode Cluster Vote")
    print("*****************************************************")
    print("Model Training Time (s):   " + "{:.2f}".format(training_time))
    print("No. Iterations to Converge: {}".format(km.n_iter_))
    print("F1 Score:  " + "{:.2f}".format(f1))
    print("Accuracy:  " + "{:.2f}".format(accuracy) + "     AUC:       " + "{:.2f}".format(auc))
    print("Precision: " + "{:.2f}".format(precision) + "     Recall:    " + "{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    # different for titanic 0,1
    plot_confusion_matrix(cm, classes=["0", "1"], title='Confusion Matrix')
    #plt.show()


def run_EM(X, y, title):
    # kdist =  [2,3,4,5]
    # kdist = list(range(2,51))
    kdist = list(np.arange(2, 50, 1))
    sil_scores = [];
    f1_scores = [];
    homo_scores = [];
    train_times = [];
    aic_scores = [];
    bic_scores = []

    for k in kdist:
        start_time = timeit.default_timer()
        em = EM(n_components=k, covariance_type='diag', n_init=1, warm_start=True, random_state=42).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)

        labels = em.predict(X)
        sil_scores.append(silhouette_score(X, labels))
        y_mode_vote = cluster_predictions(y, labels)
#        f1_scores.append(f1_score(y, y_mode_vote))
        homo_scores.append(homogeneity_score(y, labels))
        aic_scores.append(em.aic(X))
        bic_scores.append(em.bic(X))

    # elbow curve for silhouette score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, sil_scores)
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score  for EM: ' + title)
    plt.savefig('em-' + title + '-silhouette.png')

    # plot homogeneity scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, homo_scores)
    plt.grid(True)
    plt.xlabel('No. Distributions')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Scores EM: ' + title)
#    plt.show()

    # plot f1 scores
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(kdist, f1_scores)
#    plt.grid(True)
#    plt.xlabel('No. Distributions')
#    plt.ylabel('F1 Score')
#    plt.title('F1 Scores EM: ' + title)
#    plt.show()

    # plot model AIC and BIC
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, aic_scores, label='AIC')
    ax.plot(kdist, bic_scores, label='BIC')
    plt.grid(True)
    plt.xlabel('# Distributions')
    plt.ylabel('Model Complexity Score')
    plt.title('EM Model Complexity: ' + title)
    plt.legend(loc="best")
#    plt.show()
    plt.savefig('em-' + title + '-aicbic.png')

if __name__ == '__main__':
    X, Y = loader.get_titantic_data()
    X_train, X_test, y_train, y_test = loader.split_data(X, Y, test_size=.3)

    titanic_df = pd.DataFrame(data=X_train,columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title"])
    survived_df = pd.DataFrame(data=y_train,columns=["Survived"])
#    seaborn.pairplot(titanic_df,hue="Survived")
#    plt.savefig('titanic-pairplot.png')

    run_kmeans(X_train,y_train.ravel(),'titanic')
    km = KMeans(n_clusters=5, random_state=42)

    evaluate_kmeans(km, X_train, y_train.ravel())
    run_EM(X_train,y_train.ravel(),'titanic')

    titanic_cluster_df = pd.DataFrame(data=pd.to_numeric(km.labels_),columns=["Cluster"])
    titanic_kmeans_df = pd.concat([titanic_df,survived_df,titanic_cluster_df],axis=1)
    plot_colors = { 0 : "red", 1 : "blue", 2 : "green", 3 : "yellow", 4: "orange", 5 : "purple", 6: "black"}
# Age -> Fare
# Passenger Class -> Sex
    age_fare_df = pd.concat([titanic_df[["Age","Fare"]],titanic_cluster_df],axis=1)
    pclass_sex_df = pd.concat([titanic_df[["Pclass","Sex"]],titanic_cluster_df],axis=1)
    seaborn.pairplot(age_fare_df,hue="Cluster",palette=plot_colors)
    plt.savefig('kmeans-titantic-age-fare.png')
    seaborn.pairplot(pclass_sex_df,hue="Cluster",palette=plot_colors)
    plt.savefig('kmeans-titantic-pclass-sex.png')

    #    seaborn.pairplot(titanic_kmeans_df,hue="Cluster",palette=plot_colors)
#    plt.savefig('kmeans-titanic-pairplot.png')

    em = EM(n_components=5, covariance_type='diag', n_init=1, warm_start=True, random_state=42).fit(X_train)
    labels = em.predict(X_train)
    titanic_cluster_df = pd.DataFrame(data=pd.to_numeric(labels),columns=["Cluster"])
    titanic_em_df = pd.concat([titanic_df,survived_df,titanic_cluster_df],axis=1)
#    seaborn.pairplot(titanic_em_df,hue="Cluster",palette=plot_colors)
#    plt.savefig('em-titanic-pairplot.png')
    age_fare_df = pd.concat([titanic_df[["Age","Fare"]],titanic_cluster_df],axis=1)
    pclass_sex_df = pd.concat([titanic_df[["Pclass","Sex"]],titanic_cluster_df],axis=1)
    seaborn.pairplot(age_fare_df,hue="Cluster",palette=plot_colors)
    plt.savefig('em-titantic-age-fare.png')
    seaborn.pairplot(pclass_sex_df,hue="Cluster",palette=plot_colors)
    plt.savefig('em-titantic-pclass-sex.png')

    X, Y = loader.get_wine_data()
    X_train, X_test, y_train, y_test = loader.split_data(X, Y, test_size=.3,scale_data=True)
    run_kmeans(X_train,y_train.ravel(),'wine')

    km = KMeans(n_clusters=5, random_state=42)
    km.fit(X_train, y_train.ravel())
    wine_data_df = DataLoader.get_wine_data_as_dataframe(X_train,y_train)
    wine_data_cluster_df = pd.DataFrame(data=pd.to_numeric(km.labels_), columns=["Cluster"])
    wine_kmeans_data_df = pd.concat([wine_data_df, wine_data_cluster_df], axis=1)
    plot_colors = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "orange", 5: "purple", 6: "black"}
    seaborn.pairplot(wine_kmeans_data_df,hue="Cluster",palette=plot_colors)
    plt.savefig('winedata-pairplot.png')

    run_EM(X_train,y_train.ravel(),'wine')
    em = EM(n_components=5, covariance_type='diag', n_init=1, warm_start=True, random_state=42).fit(X_train)
    labels = em.predict(X_train)
    wine_data_cluster_df = pd.DataFrame(data=pd.to_numeric(labels),columns=["Cluster"])
    wine_data_em_df = pd.concat([wine_data_df,wine_data_cluster_df],axis=1)
    seaborn.pairplot(wine_data_em_df,hue="Cluster",palette=plot_colors)
    plt.savefig('em-winedata-pairplot.png')

