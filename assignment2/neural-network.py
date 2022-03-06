import DataLoader as loader
from sklearn.metrics import accuracy_score
import mlrose_hiive as mlrose
import time

# Load Data
X, Y = loader.get_titantic_data()
x_train, x_test, y_train, y_test = loader.split_data(X, Y, test_size=0.3, scale_data=True)

# Setup parameters (as close as possible to ones in Assignment 1)
params = { "hidden_nodes" : [100],
           "activation" : "relu",
           "max_iters" : 4000,
           "early_stopping" : False,
           "max_attempts" : 100,
           "random_state" : 42
        }

# Specify algorithms that we are going to create Neural Networks on
algorithms = ['random_hill_climb','simulated_annealing','genetic_alg']

# Perform experiment for each algorithm
for algorithm in algorithms:
    model = mlrose.NeuralNetwork(algorithm=algorithm,**params)
    t = time.time()
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    y_train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Training accuracy for {}: {}".format(algorithm, y_train_accuracy))

    y_test_pred = model.predict(x_test)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy for {}: {}".format(algorithm, y_test_accuracy))
    print("Elapsed time: {}".format(time.time()-t))