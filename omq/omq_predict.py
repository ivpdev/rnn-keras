import numpy
import sys

import omq_model_def
import omq_prepare

import os
print(os.path.dirname(os.path.realpath(__file__)))

model = omq_model_def.get_model(omq_prepare.X, omq_prepare.y)

# load the network weights
filename = "./data/model/v4weights-improvement-03-6.1756.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_word = dict((i, c) for i, c in enumerate(omq_prepare.words))
print("int_to_word")
print(int_to_word[0])

# pick a random seed
start = numpy.random.randint(0, len(omq_prepare.X_int)-1)
pattern = omq_prepare.X_int[start]
print("@1")
print(pattern)

print("Seed:")
print("\"", ''.join([int_to_word[value] for value in pattern]), "\"")

generated=""
# generate characters
for i in range(10):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(omq_prepare.n_vocab)
   # print("@2")
   # print(x)

    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_word[index]
    seq_in = [int_to_word[value] for value in pattern]
    sys.stdout.write(',')
    generated += result
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")
print(generated)