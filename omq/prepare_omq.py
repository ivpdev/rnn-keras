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

def delete_newlines(text):
    return text.replace('\n', ' ')

def build_char_to_int(text):
    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    return char_to_int, len(chars)

def encode_char(char, dict):
    pass

def encode_string(char, dict):
    pass

categories, interactions = read_data()
interaction_texts = seq(interactions).map(to_request_row).map(lambda i: i['text_raw']).map(delete_newlines).to_list()

X_text, y_text = generate_training_data(interaction_texts)

char_to_int, n_vocab = build_char_to_int(' '.join(interaction_texts))
X_int = list(map(lambda x: [char_to_int[char] for char in x], X_text))
y_int = list(map(lambda y1: char_to_int[y1], y_text))

n_patterns = len(X_int)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(X_int, (n_patterns, hyper_params['seq_length'], 1))

X = X / float(n_vocab)

y = np_utils.to_categorical(y_int)


#TODO remove special chars ?
#TODO to lower?

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')



from keras.callbacks import ModelCheckpoint

# define the checkpoint
checkpoint_filepath="data/model/v3weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print('Fitting model....')
# prepare.model.fit(prepare.X[0:1000], prepare.y[0:1000], epochs=10, batch_size=128, callbacks=callbacks_list)
prepare.model.fit(X, y, epochs=10, batch_size=32, callbacks=callbacks_list)


exit()


