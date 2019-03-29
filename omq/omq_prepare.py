import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

import xml.etree.ElementTree
from functional import seq

import omq_model_def

import nltk
nltk.download('punkt')

hyper_params = {
    'seq_length_chars': 10,
    'seq_length_words': 3,
    'empty_char': '\t'
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

def generate_seqs_from_text_chars(text):
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

def generate_seqs_from_words(words):
    dataX = []
    dataY = []
    n_words = len(words)
    seq_length = hyper_params['seq_length_words']
    empty_char = '\t'

    #TODO check if input is shorter then seq_length
    for i in range(0, n_words - 1, 1):
        if (i < (n_words - seq_length)):

            seq_in = words[i:i + seq_length]
            seq_out = words[i + seq_length]
        else:
            seq_in = words[i:n_words] + [empty_char for j in range(0, (seq_length - (n_words - i)))] #(empty_char * (seq_length - (n_words - i)))
            seq_out = empty_char

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

def generate_training_data_words(texts):
    X = []
    y = []
    for text in texts:
        X1, y1 = generate_seqs_from_words(text)
        X.extend(X1)
        y.extend(y1)

    return X, y

def delete_newlines(text):
    return text.replace('\n', ' ')

def build_char_to_int(text):
    chars = sorted(list(set(text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    return char_to_int, len(chars), chars

def build_word_to_int(words):
    chars = sorted(list(set(words)))
    word_to_int = dict((c, i) for i, c in enumerate(chars))

    return word_to_int, len(words)

def encode_char(char, dict):
    pass

def encode_string(char, dict):
    pass

categories, interactions = read_data()
# interaction_texts = seq(interactions).map(to_request_row).map(lambda i: i['text_raw']).map(delete_newlines).to_list()

interaction_texts_seq = seq(interactions).map(to_request_row).map(lambda i: i['text_raw']).map(delete_newlines)

words = interaction_texts_seq.flat_map(nltk.word_tokenize).to_list()

tokenized_interaction_texts = interaction_texts_seq.map(nltk.word_tokenize).to_list()

# X_text, y_text = generate_training_data(interaction_texts)
X_text, y_text = generate_training_data_words(tokenized_interaction_texts)

all_words = seq(tokenized_interaction_texts).flat_map(lambda x:x).to_list()
empty_char = hyper_params['empty_char']
all_words.append(empty_char)

#char_to_int, n_vocab, chars = build_char_to_int('\t'.join(interaction_texts))
word_to_int, n_vocab = build_word_to_int(all_words)

#X_int = list(map(lambda x: [char_to_int[char] for char in x], X_text))
#y_int = list(map(lambda y1: char_to_int[y1], y_text))

X_int = list(map(lambda x: [word_to_int[word] for word in x], X_text))
y_int = list(map(lambda y1: word_to_int[y1], y_text))

n_patterns = len(X_int)

# reshape X to be [samples, time steps, features]
# X = numpy.reshape(X_int, (n_patterns, hyper_params['seq_length'], 1))
X = numpy.reshape(X_int, (n_patterns, hyper_params['seq_length_words'], 1))

X = X / float(n_vocab)

y = np_utils.to_categorical(y_int)


#TODO remove special chars ?
#TODO to lower?

model = omq_model_def.get_model(X, y)

from keras.callbacks import ModelCheckpoint

# define the checkpoint
checkpoint_filepath="data/model/v3weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print('Fitting model....')
# prepare.model.fit(prepare.X[0:1000], prepare.y[0:1000], epochs=10, batch_size=128, callbacks=callbacks_list)
#model.fit(X, y, epochs=10, batch_size=32, callbacks=callbacks_list)


