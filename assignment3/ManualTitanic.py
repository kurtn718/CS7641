import numpy as np
import pandas as pd
from sklearn import metrics

import DataLoader as loader

X, Y = loader.get_titantic_data()
X_train, X_test, y_train, y_test = loader.split_data(X,Y,test_size=.3)

y_predict = np.zeros((len(X_test),1))
row_index = 0
for item in X_test:
    if item[1] == 1:
        y_predict[row_index] = 1
    if item[2] < 18:
        y_predict[row_index] = 1
    row_index = row_index + 1

print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
