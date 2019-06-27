import pickle
import os
import numpy as np

#All of your data must be in this folder, MUST CHANGE IT
data_path = "/Users/Alex Giacobbi/Desktop/ETS/"
#For each video, we have several labels, label 0 is the hirability score, so we are using that.
target_label_index = 0

#Just a helper function for loading a .pkl file.
def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

#Given the name of the file, it will return all data relevant to that file assuming that the function has access 
#to the variables declared above
def read_a_file(file_name):
    text_data = np.array(text[file_name]['features'])
    visual_data = np.array(visual[file_name]['features'])
    acoustic_data = np.array(acoustic[file_name]['features'])
    y_label_data = np.array([y_labels["labels"][file_name][target_label_index]])
    #print(text_data, visual_data,acoustic_data, y_label_data)
    return text_data, acoustic_data, visual_data, y_label_data


def flattenSentence(sentIn, maxWords, dimension):
    sent = np.array(sentIn)
    
    if (sent.shape[0] >= maxWords):
        words = sent[:maxWords].flatten()
    else:
        sent = np.append(sent, np.zeros((maxWords - sent.shape[0], dimension)))
        words = sent.flatten()
        
    return words


def flattenAllText(sentMatrix, maxSents, maxWords, dimension):
    
    if (sentMatrix.shape[0] >= maxSents):
        return np.array([flattenSentence(sentMatrix[i], maxWords, dimension) for i in range(maxSents)]).flatten()
    else:
        s = np.array([flattenSentence(i, maxWords, dimension) for i in sentMatrix]).flatten()
        s = np.append(s, np.zeros(((maxSents - sentMatrix.shape[0]) * dimension * maxWords)))
        return s



def main():
    #The revised_id_list.pkl contains the name of all the video files arranged in three groups:train,dev,test 
    dataset_id_file = os.path.join(data_path, "revised_id_list.pkl")
    dataset_id = load_pickle(dataset_id_file)
    train = dataset_id['train']
    dev = dataset_id['dev']
    test = dataset_id['test']
    #print(train)
    #print(dev)
    #print(test)

    #Some important info
    #Text data dimension
    d_text = 300
    #audio
    d_acoustic=81
    #video
    d_visual=35

    #Then facet_file contains all the visual data
    facet_file = os.path.join(data_path, 'revised_facet.pkl')
    #All the acoustic data
    covarep_file = os.path.join(data_path, "covarep.pkl")
    #All the text data. Instead of simple text, we are using 300D golve repressentation of a word
    word_vec_file = os.path.join(data_path, "glove_vectors.pkl")
    #These are the labels corresponding to each video
    y_labels = os.path.join(data_path, "video_labels.pkl")

    #Now we need to load the files
    visual = load_pickle(facet_file)
    acoustic = load_pickle(covarep_file)
    text = load_pickle(word_vec_file)
    y_labels = load_pickle(y_labels)

    trainMatrix = makeMatrix(train)
    devMatrix = makeMatrix(dev)
    testMatrix = makeMatrix(test)

    print(trainMatrix.shape)
    print(devMatrix.shape)
    print(testMatrix.shape)

    with open(os.path.join(data_path, 'trainData.pkl'), 'wb') as fout:
        pickle.dump(trainMatrix, fout)
        
    with open(os.path.join(data_path, 'devData.pkl'), 'wb') as fout:
        pickle.dump(devMatrix, fout)
        
    with open(os.path.join(data_path, 'testData.pkl'), 'wb') as fout:
        pickle.dump(testMatrix, fout)

main()