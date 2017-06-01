import prepare

from keras.callbacks import ModelCheckpoint

# define the checkpoint
checkpoint_filepath="data/model/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print 'Fitting model....'
prepare.model.fit(prepare.X[0:1000], prepare.y[0:1000], epochs=10, batch_size=128, callbacks=callbacks_list)
