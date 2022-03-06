import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_titantic_data():
    titanicX = pd.read_csv("titanicX.csv")
    titanicY = pd.read_csv("titanicY.csv")
    return (titanicX.values,titanicY.values)

def split_data(X,Y,test_size=.30,scale_data=False):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,random_state=1)

    if scale_data:
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

    return (X_train, X_test, y_train, y_test)

def get_wine_data():
    data = pd.read_csv('wine.data', header=None)
    data.columns = ['Label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash ', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']
    values = data.values
    X = values[:, 1:13]
    Y = values[:, 0]
    print(len(Y))
    return (X, Y)
