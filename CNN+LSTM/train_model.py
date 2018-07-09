import caption_generator
from keras.callbacks import ModelCheckpoint

def train_model(weight = None, batch_size=256, epochs = 10):

    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()
    model.summary()
    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = 'weights-improvement-{epoch:02d}-{count:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=5, verbose=1, callbacks=callbacks_list)
    #model.fit_generator(cg.data_generator(batch_size=batch_size), epochs=1, verbose=1, callbacks=callbacks_list)
    counter = 1

    model.optimizer.lr = 0.008
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=5, verbose=1, callbacks=callbacks_list)
    counter = 2
    model.optimizer.lr = 0.006
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=5, verbose=1, callbacks=callbacks_list)
    
    counter = 4
    model.optimizer.lr = 0.004
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=5, verbose=1, callbacks=callbacks_list)

    counter = 5
    model.optimizer.lr = 0.002
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=5, verbose=1, callbacks=callbacks_list)

    counter = 6
    model.optimizer.lr = 0.001
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=45, verbose=1, callbacks=callbacks_list)

    try:
        model.save('/home/manish.singhal/Image-Captioning-master/caption_generator/Models/WholeModel.h5', overwrite=True)
        model.save_weights('/home/manish.singhal/Image-Captioning-master/caption_generator/Models/Weights.h5',overwrite=True)
    except:
        print ("Error in saving model.")
    print ("Training complete...\n")

if __name__ == '__main__':
    train_model(epochs=50)
