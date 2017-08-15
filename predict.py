import prepare
import numpy
import sys

# load the network weights
filename = "data/model/v2/v2weights-improvement-09-2.0920.hdf5"
prepare.model.load_weights(filename)
prepare.model.compile(loss='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i, c) for i, c in enumerate(prepare.chars))

# pick a random seed
start = numpy.random.randint(0, len(prepare.dataX)-1)
pattern = prepare.dataX[start]
print "Seed:"
print "\"", ''.join([int_to_char[value] for value in pattern]), "\""

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

print "\nDone."
print generated