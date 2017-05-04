from id3 import Id3Estimator
from id3.data.load_data import load_data
from sklearn.datasets import load_breast_cancer
from id3.export import export_graphviz
import time


def run():
    X, y, targets = load_data("glass.arff")
    print(targets)
    #bunch = load_breast_cancer()
    #X, y, targets = bunch.data, bunch.target, bunch.feature_names
    id3Estimator = Id3Estimator(prune=False)
    t = time.time()
    id3Estimator.fit(X, y)
    print("Model done: {}".format(time.time() - t))
    export_graphviz(id3Estimator.tree_, "test.dot", feature_names=targets)
    print(len(id3Estimator.tree_.classification_nodes))
    print(len(id3Estimator.tree_.classification_nodes) + len(id3Estimator.tree_.feature_nodes))
    id3Estimator.predict(X)


if __name__ == '__main__':
    run()
