import sklearn.model_selection as ms
import time
from plotter import *
import DataLoader as loader
from sklearn.tree import DecisionTreeClassifier

def plot_training_curve_for_model(model,training_x,training_y,classifier_type,dataset_name):
    train_sizes = np.append(np.linspace(0.01, 0.1, 9, endpoint=False),
                            np.linspace(0.1, 1, 10, endpoint=True))

    train_sizes, train_scores, test_scores = ms.learning_curve(
            model,
            training_x,
            training_y.ravel(),
            cv=5,
            train_sizes=train_sizes,
            n_jobs=1)

    curve_train_scores = pd.DataFrame(index=train_sizes, data=train_scores)
    curve_test_scores = pd.DataFrame(index=train_sizes, data=test_scores)

    plot = plot_learning_curve('Learning Curve: {} - {}'.format(classifier_type, dataset_name),
                              train_sizes,
                              train_scores, test_scores)
    plot.savefig('{}_{}_LearningCurve.png'.format(classifier_type, dataset_name), format='png', dpi=150)


def plot_timing_curve_for_model(model, x, y,classifier_name,dataset_name, seed=42):
    sizes = np.linspace(0.1, 0.9, 9, endpoint=True)
    num_tests = 4
    output = dict()
    output['train'] = np.zeros(shape=(len(sizes), num_tests))
    output['test'] = np.zeros(shape=(len(sizes), num_tests))
    for i, train_size_pct in enumerate(sizes):
        for j in range(num_tests):
            np.random.seed(seed)
            x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=1 - train_size_pct, random_state=seed)
            start_time = time.perf_counter()
            model.fit(x_train, y_train.ravel())
            output['train'][i, j] = (time.perf_counter() - start_time)
            start_time = time.perf_counter()
            model.predict(x_test)
            output['test'][i, j] = (time.perf_counter() - start_time)

    training_df = pd.DataFrame(output['train'], index=sizes)
    testing_df = pd.DataFrame(output['test'], index=sizes)
    plot = plot_model_timing('{} - {}'.format(classifier_name, dataset_name),
                            np.array(sizes) * 100, training_df, testing_df)
    plot.savefig('{}_{}_TimingCurve.png'.format(classifier_name, dataset_name), format='png', dpi=150)

def plotExperiment(model,x_y_data,classifier_name,dataset_name,scale_data=False,test_size=0.3):
    X, Y = x_y_data
    x_train, x_test, y_train, y_test = loader.split_data(X,Y,test_size=test_size,scale_data=scale_data)
    plot_timing_curve_for_model(model,X,Y,classifier_name,dataset_name)

    plot_training_curve_for_model(model,
                                 x_train,
                                  y_train,
                                  classifier_name,
                                  dataset_name)
if __name__ == '__main__':
    model = DecisionTreeClassifier(criterion="entropy",min_samples_split=25)
    plotExperiment(model,loader.get_titantic_data(),'Decision Tree','Titanic')
