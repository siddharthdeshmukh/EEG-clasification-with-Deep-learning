from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import numpy as np
import os.path as op
import os
import glob

print(__doc__)



def load_data(user,type_folder,train):

    tmin, tmax = -1., 4.
    event_id = dict(LEFT_HAND=2, RIGHT_HAND=3)
    raw_edf = []
    path = op.join('data_i2r',user)
    directories = os.listdir(path)
    for data_folder in directories:
        file_list = glob.glob('data_i2r' + '/' + user + '/' + data_folder + '/' + type_folder + '/*.edf')

        raw_files = [read_raw_edf(raw_fnames, preload=True, stim_channel='auto')for raw_fnames in file_list]
        raw_edf.extend(raw_files)

    raw = concatenate_raws(raw_edf)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(4., 40., fir_design='firwin', skip_by_annotation='edge')   # 4-40Hz

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
    print events[:10]

    # Read epochs (train will be done only between 0.5 and 2.5s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
    tmax =2.5
    tmin = 0.5
    epochs_train = []
    while (tmax<4.1):
        epochs_train.append(epochs.copy().crop(tmin=tmin, tmax=tmax))
        tmin=tmin+0.1
        tmax=tmax+0.1

    labels = [epochs_from_train.events[:, -1] - 2 for epochs_from_train in epochs_train]
    labels_array = np.array(labels)
    epochs_data = epochs.get_data()
    epochs_data_train = [epochs_from_train.get_data() for epochs_from_train in epochs_train]
    epochs_array_train = np.array(epochs_data_train)
    #split data into training and testing set
    X = []
    y = []
    i=0
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    if (len(epochs_data_train) != len(labels_array)):
        print "Something is not right"
    else:
        while (i<len(epochs_array_train)):
            X.extend(epochs_array_train[i])
            y.extend(labels_array[i])
            # cv_split = cv.split(epochs_data_train[i])
            # for train_idx, test_idx in cv_split:
            #     #X_train, X_test = epochs_data_train[train_idx],epochs_data_train[test_idx]
            #     #y_train, y_test = labels[train_idx], labels[test_idx]
            #     X_train.append(epochs_data_train[i][train_idx])
            #     X_test.append(epochs_data_train[i][test_idx])
            #     y_train.append(labels_array[i][train_idx])
            #     y_test.append(labels_array[i][test_idx])
            i=i+1
    return np.array(X),np.array(y)


# if __name__ == '__main__':
#     data_directory = 'data_i2r';
#     user = 'subject1'
#     (X_train,y_train,X_test,y_test)=load_data(user)
#     print ("train data size is " + str(X_train.size))
#     print ("test data size is  "+ str(X_test.size))