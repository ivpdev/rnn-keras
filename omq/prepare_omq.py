import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

import xml.etree.ElementTree
from functional import seq

hyper_params = {
    'seq_length': 100
}

def read_data():
    categories_iter = xml.etree.ElementTree.parse('./data/OMQ/omq_public_categories.xml').getroot().iter('category')
    interactions_root = xml.etree.ElementTree.parse('./data/OMQ/omq_public_interactions.xml').getiterator('interaction')

    return categories_iter, interactions_root


def to_request_row(request_element):
    text = request_element.findtext('text/relevantText').strip()
    category = request_element.findtext('metadata/category')
    id = request_element.findtext('metadata/id')

    return {'id': id, 'category': category, 'text_raw': text }

def generate_seqs_from_text(text):
    dataX = []
    dataY = []
    n_chars = len(text)
    seq_length = hyper_params['seq_length']
    char_end_of_text = '\t'

    #TODO check if input is shorter then seq_length

    for i in range(0, n_chars - 1, 1):
        if (i < (n_chars - seq_length)):
            seq_in = text[i:i + seq_length]
            seq_out = text[i + seq_length]
        else:
            seq_in = text[i:n_chars] + (char_end_of_text * (seq_length - (n_chars - i)))
            seq_out = char_end_of_text

        dataX.append(seq_in)
        dataY.append(seq_out)

    return dataX, dataY

def generate_training_data(texts):
    X = []
    y = []
    for text in texts:
        X1, y1 = generate_seqs_from_text(text)
        X.extend(X1)
        y.extend(y1)

    return X, y

def build_char_to_int(text):
    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    return char_to_int

def encode_char(char, dict):
    pass

def encode_string(char, dict):

    pass

categories, interactions = read_data()
interaction_texts = seq(interactions).map(to_request_row).map(lambda i: i['text_raw']).to_list()
#TODO replace new line with space

X_text, y_text = generate_training_data(interaction_texts)




# filename = "data/omq_interactions_text.txt"
#raw_text = open(filename).read()

raw_text = texts = '\n'.join(interaction_texts)
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
print(chars)
#TODO remove special chars ?

char_to_int = dict((c, i) for i, c in enumerate(chars))
print(char_to_int)


exit()


n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

#print "Preview patterns:"
#print dataX[0:10]
#print '--------'
#print dataY[0:10]


# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#print '-----------'
#print X
#print '-----------'
#print y

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

