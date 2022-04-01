from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics.pairwise import pairwise_distances
from itertools import product
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoader as loader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM
import seaborn
import DataLoader

def performPCAExperiment(X, y, title):
    pca = PCA(random_state=42).fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Cum. Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'r-')
    ax2.set_ylabel('Eigenvalues', color='r')
    ax2.tick_params('y', colors='r')
    plt.grid(False)

    plt.title("PCA Eigenvalues and Explained Variance on " + title)
    fig.tight_layout()
    plt.savefig('pca-' + title + '.png')

    print(title + " Eigen Values ")
    print(pca.singular_values_)
    print(title + " Cumulative Sum")
    print(cum_var)

def performICAExperiment(X, y, title):
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    ica = ICA(random_state=42,max_iter=1000,tol=1e-3)
    kurtosis = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurtosis.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurtosis, 'b-')
    plt.grid(False)
    plt.savefig('ica-' + title + '.png')

    print("Kurtosis:")
    print(kurtosis)

def pairwiseDistCorr(X1, X2):
    assert X1.shape[0] == X2.shape[0]

    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def performRCAExperiement(X, y, title):
    dims = list(np.arange(2, (X.shape[1] - 1), 3))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)

    for i, dim in product(range(5), dims):
        rp = RCA(random_state=i, n_components=dim)
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()

    fig, ax1 = plt.subplots()
    ax1.plot(dims, mean_recon, 'b-')
    ax1.set_xlabel('Random Components')
    ax1.set_ylabel('Mean Reconstruction Correlation', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(dims, std_recon, 'r-')
    ax2.set_ylabel('STD Reconstruction Correlation', color='r')
    ax2.tick_params('y', colors='r')
    plt.grid(False)

    plt.title("Random Components: " + title)
    fig.tight_layout()
    plt.savefig('rca-' + title + '.png')


def run_RFC(X, y, df_original):
    rfc = RFC(n_estimators=500, min_samples_leaf=round(len(X) * .01), random_state=5, n_jobs=-1)
    imp = rfc.fit(X, y).feature_importances_
    imp = pd.DataFrame(imp, columns=['Feature Importance'], index=df_original.columns[2::])
    imp.sort_values(by=['Feature Importance'], inplace=True, ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    imp = imp[imp['Cum Sum'] <= 0.95]
    top_cols = imp.index.tolist()
    return imp, top_cols

def performTitanicKMeansDimReduction(x_train,y_train,x_train_orig,algorithm="None"):
    km = KMeans(n_clusters=5, random_state=42)
    km.fit(x_train, y_train)

    titanic_df = pd.DataFrame(data=x_train_orig,columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title"])
    survived_df = pd.DataFrame(data=y_train,columns=["Survived"])

    titanic_cluster_df = pd.DataFrame(data=pd.to_numeric(km.labels_), columns=["Cluster"])
    titanic_kmeans_df = pd.concat([titanic_df, survived_df, titanic_cluster_df], axis=1)
    plot_colors = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "orange", 5: "purple", 6: "black"}
    # Age -> Fare
    # Passenger Class -> Sex
    age_fare_df = pd.concat([titanic_df[["Age", "Fare"]], titanic_cluster_df], axis=1)
    pclass_sex_df = pd.concat([titanic_df[["Pclass", "Sex"]], titanic_cluster_df], axis=1)
    seaborn.pairplot(age_fare_df, hue="Cluster", palette=plot_colors)
    plt.savefig('kmeans-titantic-' + algorithm + '-age-fare.png')
    seaborn.pairplot(pclass_sex_df, hue="Cluster", palette=plot_colors)
    plt.savefig('kmeans-titantic-' + algorithm + '-pclass-sex.png')

    seaborn.pairplot(titanic_kmeans_df,hue="Cluster",palette=plot_colors)
    plt.savefig('kmeans-titanic-' + algorithm + '-pairplot.png')

def performTitanicEMDimReduction(x_train,y_train,x_train_orig,algorithm="None"):
    titanic_df = pd.DataFrame(data=x_train_orig,columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","Title"])
    survived_df = pd.DataFrame(data=y_train,columns=["Survived"])

    em = EM(n_components=5, covariance_type='diag', n_init=1, warm_start=True, random_state=42).fit(x_train)
    labels = em.predict(x_train)
    titanic_cluster_df = pd.DataFrame(data=pd.to_numeric(labels), columns=["Cluster"])
    titanic_em_df = pd.concat([titanic_df, survived_df, titanic_cluster_df], axis=1)
    plot_colors = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "orange", 5: "purple", 6: "black"}
    seaborn.pairplot(titanic_em_df,hue="Cluster",palette=plot_colors)
    plt.savefig('em-titanic-' + algorithm + '-pairplot.png')

    # Age -> Fare
    # Passenger Class -> Sex
    age_fare_df = pd.concat([titanic_df[["Age", "Fare"]], titanic_cluster_df], axis=1)
    pclass_sex_df = pd.concat([titanic_df[["Pclass", "Sex"]], titanic_cluster_df], axis=1)
    seaborn.pairplot(age_fare_df, hue="Cluster", palette=plot_colors)
    plt.savefig('em-titantic-' + algorithm + '-age-fare.png')
    seaborn.pairplot(pclass_sex_df, hue="Cluster", palette=plot_colors)
    plt.savefig('em-titantic-' + algorithm + '-pclass-sex.png')

def performWineKMeansDimReduction(x_train,y_train,x_train_orig,algorithm="None"):
    km = KMeans(n_clusters=5, random_state=42)
    km.fit(x_train, y_train)

    wine_data_df = DataLoader.get_wine_data_as_dataframe(x_train_orig,y_train)
    wine_data_cluster_df = pd.DataFrame(data=pd.to_numeric(km.labels_), columns=["Cluster"])
    wine_kmeans_data_df = pd.concat([wine_data_df, wine_data_cluster_df], axis=1)
    plot_colors = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "orange", 5: "purple", 6: "black"}
    seaborn.pairplot(wine_kmeans_data_df,hue="Cluster",palette=plot_colors)
    plt.savefig('kmeans-winedata-' + algorithm + '-pairplot.png')


def performWineEMDimReduction(x_train,y_train,x_train_orig,algorithm="None"):
    em = EM(n_components=5, covariance_type='diag', n_init=1, warm_start=True, random_state=42).fit(x_train)
    labels = em.predict(x_train)

    wine_data_df = DataLoader.get_wine_data_as_dataframe(x_train_orig,y_train)
    wine_data_cluster_df = pd.DataFrame(data=pd.to_numeric(labels),columns=["Cluster"])
    wine_data_em_df = pd.concat([wine_data_df,wine_data_cluster_df],axis=1)
    plot_colors = {0: "red", 1: "blue", 2: "green", 3: "yellow", 4: "orange", 5: "purple", 6: "black"}
    seaborn.pairplot(wine_data_em_df,hue="Cluster",palette=plot_colors)
    plt.savefig('em-winedata-' + algorithm + '-pairplot.png')

def performTitanicDimReduction():
    X, Y = loader.get_titantic_data()
    X_train, X_test, y_train, y_test = loader.split_data(X, Y, test_size=.3,scale_data=True)
    X_train1, _, _, _ = loader.split_data(X, Y, test_size=.3,scale_data=False)

    titanic_df = pd.DataFrame(data=X_train,
                              columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize",
                                       "Title"])

    performPCAExperiment(X_train,y_train,"Titanic")
    performICAExperiment(X_train,y_train,"Titanic")
    performRCAExperiement(X_train,y_train,"Titanic")


    pca_titanic = PCA(n_components=6,random_state=42).fit_transform(X_train)
    ica_titanic = ICA(n_components=5,random_state=42).fit_transform(X_train)
    rca_titanic = RCA(n_components=5,random_state=42).fit_transform(X_train)

# Uncomment to view plots
#    performTitanicKMeansDimReduction(pca_titanic, y_train, X_train1, "PCA")
#    performTitanicKMeansDimReduction(ica_titanic, y_train, X_train1, "ICA")
#    performTitanicKMeansDimReduction(rca_titanic, y_train, X_train1, "RC")

#    performTitanicEMDimReduction(pca_titanic, y_train, X_train1, "PCA")
#    performTitanicEMDimReduction(ica_titanic, y_train, X_train1, "ICA")
#    performTitanicEMDimReduction(rca_titanic, y_train, X_train1, "RC")

def performWineDimReduction():
    X, Y = loader.get_wine_data()
    X_train, X_test, y_train, y_test = loader.split_data(X, Y, test_size=.3,scale_data=True)
    wine_data_df = loader.get_wine_data_as_dataframe(X_train,y_train)

    performPCAExperiment(X_train,y_train,"Wine")
    performICAExperiment(X_train,y_train,"Wine")
    performRCAExperiement(X_train,y_train,"Wine")

    pca_titanic = PCA(n_components=7,random_state=42).fit_transform(X_train)
    ica_titanic = ICA(n_components=5,random_state=42).fit_transform(X_train)
    rca_titanic = RCA(n_components=5,random_state=42).fit_transform(X_train)

# Uncomment to view plots
#    performWineKMeansDimReduction(pca_titanic, y_train, X_train, "PCA")
#    performWineKMeansDimReduction(ica_titanic, y_train, X_train, "ICA")
#    performWineKMeansDimReduction(rca_titanic, y_train, X_train, "RC")

#    performWineEMDimReduction(pca_titanic, y_train, X_train, "PCA")
#    performWineEMDimReduction(ica_titanic, y_train, X_train, "ICA")
#    performWineEMDimReduction(rca_titanic, y_train, X_train, "RC")


if __name__ == '__main__':
    performTitanicDimReduction()
    performWineDimReduction()
