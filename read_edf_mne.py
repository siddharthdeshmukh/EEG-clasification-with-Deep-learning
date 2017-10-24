from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
import os.path as op
import os
import glob

print(__doc__)



def load_data(user):

    tmin, tmax = -1., 4.
    event_id = dict(LEFT_HAND=2, RIGHT_HAND=3)
    raw_edf = []
    path = op.join('data_i2r',user)
    directories = os.listdir(path)
    for data_folder in directories:
        file_list = glob.glob('data_i2r' + '/' + user + '/' + data_folder + '/' + 'DataTraining' + '/*.edf')

        raw_files = [read_raw_edf(raw_fnames, preload=True, stim_channel='auto')for raw_fnames in file_list]
        raw_edf.extend(raw_files)

    raw = concatenate_raws(raw_edf)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
    print events[:10]

    # Read epochs (train will be done only between 0.5 and 2.5s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=0.5, tmax=2.5)
    labels = epochs.events[:, -1] - 2

    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    #split data into training and testing set
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    X_train =[]
    X_test =[]
    y_train = []
    y_test = []
    for train_idx, test_idx in cv_split:
        X_train, X_test = epochs_data_train[train_idx],epochs_data_train[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

    return X_train,y_train,X_test,y_test


# if __name__ == '__main__':
#     data_directory = 'data_i2r';
#     user = 'subject1'
#     (X_train,y_train,X_test,y_test)=load_data(user)
#     print ("train data size is " + str(X_train.size))
#     print ("test data size is  "+ str(X_test.size))