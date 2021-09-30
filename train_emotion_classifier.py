from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator from load_and_process import load_fer2013
from load_and_process import preprocess_input from models.cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split


# setting up the parameters batch_size = 32
num_epochs = 100
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50 base_path = 'models/'

# data generator function for generating batches of images from training data data_generator = ImageDataGenerator(
featurewise_center=False, featurewise_std_normalization=False, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=.1, horizontal_flip=True)

# defining the ML model, optimization function, loss function model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# defining the necessary callbacks
log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False) # logging the details for each epoch early_stop = EarlyStopping('val_loss', patience=patience) #stop the process if validation loss doesn't decrease after 50 epochs
#reduce the learning rate for the model by a factor of 0.1 if the validation accuracy doesn't im porove for 5 epochs
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1) trained_models_path = base_path + 'mini_Xcpetion'
model_names = trained_models_path + '.{epoch:02d}-
{val_acc:.2f}.hdf5'	#defining model name to be saved
#save model if validation accuracy improves from earlier epochs
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only= True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


# loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)	#resizing the images num_samples, num_classes = emotions.shape
#splitting the dataset into 80% training and 20% validation testing data
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True) #fitting the model on the dataset
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size), steps_per_epoch=len(xtra in) / batch_size, epochs=num_epochs, verbose=1, callbacks=callbacks, validation_data=(xtest, ytest))
