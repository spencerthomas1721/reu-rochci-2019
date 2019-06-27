import numpy as np
import pickle as pkl
import os
from sklearn import svm

data_path = "/Users/Alex Giacobbi/Desktop/ETS/"

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


def main():
    trainSet = load_pickle(os.path.join(data_path, 'trainData.pkl'))
    print(trainSet, trainSet.shape)

    devSet = load_pickle(os.path.join(data_path, 'devData.pkl'))
    print(devSet, devSet.shape)

    testSet = load_pickle(os.path.join(data_path, 'testData.pkl'))
    print(testSet, testSet.shape)

    trainFeatures = trainSet[:, :332800]
    trainLabels = trainSet[:, 332800]
    print(trainFeatures.dtype)
    print(trainLabels.dtype)

    #replace NaNs with 0s
    trainFeatures = np.nan_to_num(trainFeatures, copy=False)
    trainLabels = np.nan_to_num(trainLabels, copy=False)

    np.isnan(trainFeatures).any()

    svmModel = svm.SVR(gamma='scale')
    svmModel.fit(trainFeatures, trainLabels)

    testFeatures = testSet[:, :332800]
    testLabels = testSet[:, 332800]

    testFeatures = np.nan_to_num(testFeatures, copy=False)
    testLabels = np.nan_to_num(testLabels, copy=False)

    devFeatures = devSet[:, :332800]
    devLabels = devSet[:, 332800]

    devFeatures = np.nan_to_num(devFeatures, copy=False)
    devLabels = np.nan_to_num(devLabels, copy=False)

    svmModel.score(testFeatures, testLabels)
    svmModel.score(devFeatures, devLabels)

main()