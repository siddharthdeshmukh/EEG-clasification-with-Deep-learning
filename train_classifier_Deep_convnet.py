import numpy as np
from sklearn import svm, pipeline, base, metrics
import eegtools
import read_edf_data
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,BatchNormalization,Convolution2D,Activation,Flatten,Dropout
from keras.utils import np_utils
from keras import backend as K
import read_edf_mne
import matplotlib.pyplot as plt
from keras.utils import plot_model

'''
Training model for classification of EEG samples into motor imagery classes
'''

# Create sklearn-compatible feature extraction and classification pipeline:


def train(x_train, y_train,X_test,y_test):
    #1. Define Model.
    X_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
    Y_train = np_utils.to_categorical(y_train, 3)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    Y_test = np_utils.to_categorical(y_test, 3)
    model = Sequential()
    model.add(Convolution2D(25, (3, 1), activation='elu', input_shape=(X_train.shape[1],X_train.shape[2], X_train.shape[3])))
    #model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(Convolution2D(50, (3, 1), activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Convolution2D(100, (3, 1), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Convolution2D(200, (2, 1), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='elu'))
    model.add(BatchNormalization())
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    print model.summary()
    history =model.fit(X_train, Y_train,validation_data=(X_test,Y_test), batch_size=30, nb_epoch=1, verbose=1)
    return model,history

def evaluate_model(model,test,y_test):
    X_test = test.reshape(test.shape[0], 1, test.shape[1], test.shape[2])
    Y_test = np_utils.to_categorical(y_test, 3)
    loss,acc = model.evaluate(X_test, Y_test, verbose=0)
    print loss,acc



if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    data_directory = 'data_i2r';
    user = 'subject1'
    # get train data
    #(x_train, y_train) = read_edf_data.load_data(data_directory, user, 'DataTraining', True)
    (X_train, y_train, X_test, y_test) = read_edf_mne.load_data(user)
    # get Test data
    #(test, y_test) = read_edf_data.load_data(data_directory, user, 'DataTraining', False)
    model,history = train(X_train,y_train,X_test,y_test)
    #evaluate_model(model,test=X_test,y_test=y_test)
    print history.history.keys()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plot_model(model, to_file='model.png')