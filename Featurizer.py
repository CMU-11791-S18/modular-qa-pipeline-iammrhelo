import abc
from abc import abstractmethod
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Featurizer:
    __metaclass__ = abc.ABCMeta

    @classmethod
    def __init__(self):  #constructor for the abstract class
        pass

    #This is the abstract method that is implemented by the subclasses.
    @abstractmethod
    def getFeatureRepresentation(self, X_train, X_val):
        pass


class CountFeaturizer(Featurizer):
    def getFeatureRepresentation(self, X_train, X_val):
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        X_val_counts = count_vect.transform(X_val)
        return X_train_counts, X_val_counts


class TfidfFeaturizer(Featurizer):
    def getFeatureRepresentation(self, X_train, X_val):
        tfidf_vect = TfidfVectorizer()
        X_train_counts = tfidf_vect.fit_transform(X_train)
        X_val_counts = tfidf_vect.transform(X_val)
        return X_train_counts, X_val_counts


def builder(string):
    try:
        return getattr(sys.modules[__name__], string)
    except AttributeError:
        return None
