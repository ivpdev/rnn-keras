import numpy
import sys

import omq_model_def
import prepare_omq

# load the network weights
filename = "./data/model/all/v3weights-improvement-09-0.9317.hdf5"
omq_model_def.model.load_weights(filename)
omq_model_def.model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(prepare_omq.chars))

# pick a random seed
start = numpy.random.randint(0, len(prepare.X_int)-1)
pattern = prepare.dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

generated=""
# generate characters
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(prepare.n_vocab)
    prediction = prepare.model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(',')
    generated += result
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")
print(generated)