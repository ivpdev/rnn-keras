import omq_model_def
import omq_prepare

X = omq_prepare.X
y = omq_prepare.y

model = omq_model_def.get_model(X, y)

from keras.callbacks import ModelCheckpoint

# define the checkpoint
checkpoint_filepath="data/model/v4weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print('Fitting model....')
model.fit(X, y, epochs=10, batch_size=32, callbacks=callbacks_list)