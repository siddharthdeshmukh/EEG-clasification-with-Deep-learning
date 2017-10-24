from mne.datasets import eegbci

subjects = 1
runs = [1,2,3,4,5,6,7, 8,9,10,11, 12,13,14]  # motor imagery: hands vs feet


for subject in range(1,40,1):
    raw_fnames = eegbci.load_data(subject, runs)

