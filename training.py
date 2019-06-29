import numpy as np
import pickle as pkl
import os
from sklearn import svm

data_path = os.getcwd()

#Just a helper function for loading a .pkl file.
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pkl.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pkl.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def splitFeaturesAndLabels(data):
    features = data[:, :(data.shape[1] - 1)]
    labels = data[:, (data.shape[1] - 1)]

    return features, labels


def main():
    #load pickle files into respective matrices
    trainSet = load_pickle(os.path.join(data_path, 'trainData.pkl'))
    devSet = load_pickle(os.path.join(data_path, 'devData.pkl'))
    testSet = load_pickle(os.path.join(data_path, 'testData.pkl'))

    trainSet = np.nan_to_num(trainSet, copy=False)
    devSet = np.nan_to_num(devSet, copy=False)
    testSet = np.nan_to_num(testSet, copy=False)

    trainFeatures, trainLabels = splitFeaturesAndLabels(trainSet)
    devFeatures, devLabels = splitFeaturesAndLabels(devSet)
    testFeatures, testLabels = splitFeaturesAndLabels(testSet)

    print('training...\n')
    svmModel = svm.SVR(gamma='scale')
    svmModel.fit(trainFeatures, trainLabels)

    # cross-validate with dev
    # from sklearn.model_selection import cross_validate
    cvResults = cross_validate(svmModel, devFeatures, devLabels) # can customize scoring method
    avg_fit_time = np.mean(cv_results["fit_time"])
    avg_score_time = np.mean(cv_results["score_time"])
    avg_score = np.mean(cv_results["test_score"])

    testScore = svmModel.score(testFeatures, testLabels)
    #svmModel.score(devFeatures, devLabels)

    print("Dev results: " + cvResults)
    print("Test results: " + testScore)

main()
