import eegtools
import pyedflib
import numpy as np
import glob
from scipy import signal
import os


def load_data(data_directory, user,training,trainOrTest):
    #data_directory = 'data_i2r'
    #user = 'subject1'
   # file_list = glob.glob(data_directory + '/' + user)
    path = os.path.join(data_directory,user)
    dir = os.listdir(path)
    train = []
    y_train = []
    test = []
    y_test = []

    for data_folder in dir:
        file_list = glob.glob(data_directory + '/' + user + '/'+ data_folder + '/' +training + '/*.edf')
        for datafile in file_list:
            print(datafile)
            raw = pyedflib.EdfReader(datafile)
            annotations = raw.readAnnotations()
            channels = raw.signals_in_file
            sigbufs = np.zeros((channels, raw.getNSamples()[0]))
            for i in np.arange(raw.signals_in_file):
                sigbufs[i, :] = raw.readSignal(i)

            # create band-pass filter for the  8--30 Hz where the power change is expected
            (b, a) = signal.butter(3, np.array([4, 30]) / (raw.getSampleFrequencies()[0] / 2), 'band')
            # band-pass filter the EEG
            filt_data = signal.lfilter(b, a, sigbufs, 1)
            # extract trials
            start = []
            annot_list = zip(annotations[0],annotations[1],annotations[2])


            for i in annot_list:
                if i[2] in ['121', '122']:
                    start.append(int(i[0] * raw.getSampleFrequencies()[0]))
            # start = [float(i[0]) * d.sample_rate for i in d.annotations]
            duration = [float(i[1]) * raw.getSampleFrequencies()[0] for i in annot_list]
            offset = [0, np.min(duration)]
            x, y = offset
            x = int(x)
            y = int(y)
            offset = x, y
            # print offset
            # print len(start)
            trials, st = eegtools.featex.windows(start, offset, filt_data)
            n = len(st)
            # extract classes
            labels = [i[2] for i in annot_list]
            y = []

            for label in labels:
                if label == '121':
                    y.append(1)
                if label == '122':
                    y.append(2)

            y = y[0:n]
            if(trainOrTest):
                train.extend(trials)
                y_train.extend(y)
            elif (not trainOrTest):
                train.extend(trials)
                y_train.extend(y)

    train = np.asarray(train)
    y_train = np.asarray(y_train)

    return train, y_train

# if __name__ == '__main__':
#     data_directory = 'data_i2r';
#     user = 'subject1'
#     (train,y_train)=load_data(data_directory, user,'DataTraining',True)
#     (test,y_test) = load_data(data_directory, user,'DataTraining',False)
#     print ("train data size is "+ str(train.size))
#     print ("test data size is  "+ str(test.size))