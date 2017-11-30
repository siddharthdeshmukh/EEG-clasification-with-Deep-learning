from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
import os.path as op
import os
import glob
from sklearn.model_selection import train_test_split

from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
import os.path as op
import os
import glob

print(__doc__)



def load_data(training=False):

    tmin, tmax = -1., 4.1

    raw_edf = []
    X = []
    y = []
    X_train =[]
    X_test = []
    y_train =[]
    y_test =[]
    stim_code = dict([(32766,1),(769,2), (770,3), (771,5), (772,4),(783,6),(276,7),(277,8),(768,9),
                      (1023,10),(1072,11)])
    if training:
        path = op.join('data_i2r', 'BCI_IV_2a', 'TrainingSet')
    if not training:
        path = op.join('data_i2r', 'BCI_IV_2a', 'TestingSet')
    #directories = os.listdir(path)
    #for data_folder in directories:
    file_list = glob.glob(path + '/*.gdf')
    print file_list
    raw_files = [read_raw_edf(raw_fnames, preload=True, stim_channel='auto')for raw_fnames in file_list]
    raw_edf.extend(raw_files)
    #events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    samplin_frequency =250;


    for edf_raw in raw_edf:
        event_id = dict()
        events = find_events(edf_raw, shortest_event=0, stim_channel='STI 014')
        events_from_edf = []
        samplin_frequency=edf_raw._raw_extras[0]['max_samp']
        original_event = edf_raw.find_edf_events()
        annot_list = zip(original_event[1], original_event[4], original_event[2])
        events_from_edf.extend(annot_list)
        events_from_edf = np.array(events_from_edf)
        i = 0
        events_arr = np.zeros(events_from_edf.shape, dtype=int)
        for i_event in events_from_edf:

            index = int((float(i_event[0])) * samplin_frequency)

            events_arr[i,:] = index,0,stim_code[int(i_event[2])]
            i=i+1

        # strip channel names of "." characters
        edf_raw.rename_channels(lambda x: x.strip('.'))
        #create Event dictionary based on File
        events_in_edf = [event[2] for event in events_arr[:]]
        if(events_in_edf.__contains__(2)):
            event_id['LEFT_HAND']=2
        if (events_in_edf.__contains__(3)):
            event_id['RIGHT_HAND'] = 3
        # if (events_in_edf.__contains__(4)):
        #     event_id['FEET'] = 4
        # if (events_in_edf.__contains__(5)):
        #     event_id['IDLE'] = 5
        # Apply band-pass filter
        edf_raw.filter(4., 40., fir_design='firwin', skip_by_annotation='edge')   # 4-40Hz



        picks = pick_types(edf_raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
        print events_arr[:10]

        # Read epochs (train will be done only between 0.5 and 2.5s)
        # Testing will be done with a running classifier
        print edf_raw
        if event_id:

            epochs = Epochs(edf_raw, events_arr, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
            tmaximum =2.5
            tminimum = 0.5
            epochs_train = []
            while (tmaximum<4.1):
                epochs_train.append(epochs.copy().crop(tmin=tminimum, tmax=tmaximum))
                tminimum=tminimum+0.1
                tmaximum=tmaximum+0.1

            labels = [epochs_from_train.events[:, -1] - 2 for epochs_from_train in epochs_train]
            labels_array = np.array(labels)
            epochs_data = epochs.get_data()
            epochs_data_train = [epochs_from_train.get_data() for epochs_from_train in epochs_train]
            epochs_array_train = np.array(epochs_data_train)
            #split data into training and testing set

            i=0
            cv = ShuffleSplit(10, test_size=0.2, random_state=42)
            if (len(epochs_data_train) != len(labels_array)):
                print "Something is not right"
            else:
                while (i<len(epochs_array_train)):
                    X.extend(epochs_array_train[i])
                    y.extend(labels_array[i])

                    # for train_idx, test_idx in cv_split:
                    #     #X_train, X_test = epochs_array_train[train_idx],epochs_array_train[test_idx]
                    #     #y_train, y_test = labels[train_idx], labels[test_idx]
                    #     X_train.append(epochs_array_train[i][train_idx])
                    #     X_test.append(epochs_array_train[i][test_idx])
                    #     y_train.append(labels_array[i][train_idx])
                    #     y_test.append(labels_array[i][test_idx])
                    i=i+1
    cv_split = cv.split(X)
    X = np.array(X)
    y = np.array(y)
    # for train_idx, test_idx in cv_split:
    #     X_train.append(X[train_idx])
    #     X_test.append(X[test_idx])
    #     y_train.append(y[train_idx])
    #     y_test.append(y[test_idx])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)
    #return X,y

# if __name__ == '__main__':
#     data_directory = 'data_i2r';
#     #user = 'subject1'
#     (X_train,y_train,X_test,y_test)=load_data(training=True)
#     print ("train data size is " + str(X_train.size))
#     print ("test data size is  "+ str(X_test.size))