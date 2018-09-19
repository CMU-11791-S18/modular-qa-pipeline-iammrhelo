import abc
from abc import abstractmethod
import sys

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier as mlp


class Classifier:
    __metaclass__ = abc.ABCMeta

    @classmethod
    def __init__(self):  # constructor for the abstract class
        pass

    # This is the abstract method that is implemented by the subclasses.
    @abstractmethod
    def buildClassifier(self, X_features, Y_train):
        pass


class MultinomialNaiveBayes(Classifier):
    def buildClassifier(self, X_features, Y_train):
        clf = MultinomialNB().fit(X_features, Y_train)
        return clf


class SVMClassifier(Classifier):
    def buildClassifier(self, X_features, Y_train):
        clf = svm.LinearSVC(verbose=True).fit(X_features, Y_train)
        return clf


class MLPClassifier(Classifier):
    def buildClassifier(self, X_features, Y_train):
        clf = mlp(
            solver='sgd',
            learning_rate_init=0.1,
            max_iter=20,
            hidden_layer_sizes=(100, ),
            random_state=1,
            verbose=True)
        clf.fit(X_features, Y_train)
        return clf


def builder(string):
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError:
        return None
