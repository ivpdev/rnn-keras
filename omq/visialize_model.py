
from keras.utils import plot_model


import omq_model_def
import omq_prepare

model = omq_model_def.get_model(omq_prepare.X, omq_prepare.y)

# load the network weights
filename = "./data/model/v4weights-improvement-10-5.5914.hdf5"
model.load_weights(filename)


plot_model(model, to_file='./data/model/viz/model.png')

