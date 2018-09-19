import argparse
import csv
import json
import sys

from Retrieval import Retrieval
from Featurizer import builder as featurizerlib
from Classifier import builder as classifierlib
from Evaluator import Evaluator


class Pipeline(object):
    def __init__(self, trainFilePath, valFilePath, saveFilePath, retrievalInstance,
                 featurizerInstance, classifierInstance):

        self.saveFilePath = saveFilePath
        self.retrievalInstance = retrievalInstance
        self.featurizerInstance = featurizerInstance
        self.classifierInstance = classifierInstance
        trainfile = open(trainFilePath, 'r')
        self.trainData = json.load(trainfile)
        trainfile.close()
        valfile = open(valFilePath, 'r')
        self.valData = json.load(valfile)
        valfile.close()
        self.question_answering()

    def makeXY(self, dataQuestions):
        X = []
        Y = []
        for question in dataQuestions:

            long_snippets = self.retrievalInstance.getLongSnippets(question)
            short_snippets = self.retrievalInstance.getShortSnippets(question)

            X.append(short_snippets)
            Y.append(question['answers'][0])

        return X, Y

    def question_answering(self):
        dataset_type = self.trainData['origin']
        candidate_answers = self.trainData['candidates']
        X_train, Y_train = self.makeXY(self.trainData['questions'])
        X_val, Y_val_true = self.makeXY(self.valData['questions'])

        # featurization
        X_features_train, X_features_val = self.featurizerInstance.getFeatureRepresentation(
            X_train, X_val)
        self.clf = self.classifierInstance.buildClassifier(
            X_features_train, Y_train)

        # Prediction
        Y_val_pred = self.clf.predict(X_features_val)

        self.evaluatorInstance = Evaluator()
        a = self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
        p, r, f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)
        print("Accuracy: " + str(a))
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        print("F-measure: " + str(f))

        with open(self.saveFilePath, 'w') as fout:
            writer = csv.writer(fout)
            for tup in zip(Y_val_true, Y_val_pred):
                writer.writerow(tup)
        return Y_val_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trainFilePath', default="./dataset/quasar-s_train_formatted.json")
    parser.add_argument(
        '--valFilePath', default="./dataset/quasar-s_dev_formatted.json")
    parser.add_argument('-f', '--featurizer', default='CountFeaturizer')
    parser.add_argument('-c', '--classifier', default='MultinomialNaiveBayes')
    parser.add_argument('-s', '--saveFilePath', type=str)
    args = parser.parse_args()

    trainFilePath = args.trainFilePath
    valFilePath = args.valFilePath
    saveFilePath = args.saveFilePath
    retrievalInstance = Retrieval()
    featurizerInstance = featurizerlib(args.featurizer)()
    classifierInstance = classifierlib(args.classifier)()
    trainInstance = Pipeline(trainFilePath, valFilePath, saveFilePath, retrievalInstance,
                             featurizerInstance, classifierInstance)
