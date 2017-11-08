import numpy as np
from sklearn import svm, pipeline, base, metrics
import eegtools
import read_edf_data
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Convolution2D,Activation,Flatten,Dropout
from keras.utils import np_utils
from keras import backend as K
from keras import optimizers
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
    y_test = np_utils.to_categorical(y_test, 3)
    model = Sequential()
    model.add(Convolution2D(25, (3, 1), activation='elu', input_shape=(X_train.shape[1],X_train.shape[2], X_train.shape[3])))
    #model.add(MaxPooling2D(pool_size=(3,1)))
    model.add(Convolution2D(50, (3, 1), activation='elu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Convolution2D(100, (3, 1), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Convolution2D(200, (2, 1), activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))
    #model.add(BatchNormalization())
    opt = optimizers.SGD(lr=0.001)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    print model.summary()
    history =model.fit(X_train, Y_train,validation_data=(X_test,y_test), batch_size=100,epochs=20, verbose=1)
    return model,history

def evaluate_model(model,test,y_test):
    X_test = test.reshape(test.shape[0], 1, test.shape[1], test.shape[2])
    Y_test = np_utils.to_categorical(y_test, 2)
    y_pred = model.predict(X_test,batch_size=30)
    Y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)
    classes = ['LEFT_HAND','RIGHT_HAND']
    confusion_metric =  metrics.confusion_matrix(np.argmax(Y_test,axis=1),Y_pred,labels=[0,1])
    loss,acc = model.evaluate(X_test, Y_test,batch_size=30,verbose=0)
    plt.title('Confusion Metrics')
    ax = plt.gca()  # get the current axes
    PCM = ax.get_children()[2]  # get the mappable, the 1st and the 2nd are the x and y axes
#    plt.colorbar(PCM, ax=ax)
    tick_marks = np.arange(len(['LEFT_HAND','RIGHT_HAND']))
    plt.xticks(tick_marks,classes)
    plt.yticks(tick_marks,classes)
    plt.tight_layout()
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.imshow(confusion_metric)
    plt.show()
    print ("Loss is %f ", loss)
    print("Accuracy is %f ", acc)



if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    data_directory = 'data_i2r';
    user = 'subject1'
    # get train data
    # (X_train, y_train) = read_edf_data.load_data(data_directory, user, 'DataTraining', True)
    (X_train, y_train) = read_edf_mne.load_data(user, 'DataTraining', train=True)
    (X_test, y_test) = read_edf_mne.load_data(user, 'DataTesting', train=False)
    # get Test data
    # (X_test, y_test) = read_edf_data.load_data(data_directory, user, 'DataTesting', False)
    model, history = train(X_train, y_train, X_test, y_test)
    evaluate_model(model, test=X_test, y_test=y_test)
    print history.history.keys()
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plot_model(model, to_file='model_deep.png')