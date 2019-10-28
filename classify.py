import os
import json
import numpy as np
import pandas as pd
from data_loader import DataLoader
from config import get_args


from scipy.ndimage.morphology import binary_fill_holes as bf
from skimage.filters import threshold_otsu, gaussian, apply_hysteresis_threshold

from skimage.transform import probabilistic_hough_line, radon, iradon, rescale
from skimage.feature import canny

from scipy.signal import find_peaks

'''
This script classifies images in the RadCure dataset which contain 'strong'
metal artifact streaks.

The program processes each patient slice-by-slice, and picks out slices with
where there is very strong evidence of artifacts.
'''

class Classifier(object):
    """ Class containing functions to run automatic (non-DL) per-slice
        classification of images with artifacts."""
    def __init__(self, args, data_loader):
        # super(Classifier, self).__init__()
        self.data_loader = data_loader

        # Test mode. If true, do only a few iterations
        self.test_mode = args.test

        self.log_txt = os.path.join(args.logdir, "preds.txt")
        # self.log_txt = os.path.join(args.logdir, "preds.json")
        self.log_csv = os.path.join(args.logdir, "preds.csv")

        self.sigma = 10 # Width of Gaussian for blur
        self.t1 = 0.01  # Threshold for removal of body
        self.t2 = 0.02  # Threshold after body is removed

    def save_patient(self, patient_id, indices) :
        ''' Save the indices which contain artifacts for one patient.
        Save to JSON file.'''
        ### SAVE SPECIFIC SLICE INDICES CONTAINING ARTIFACTS ###
        # JSON implementation
        # with open(self.log_txt, 'r') as f :           # Open and read entire JSON
        #     old_dict = json.load(f)                   # Unencode JSON as a dict
        #
        # old_dict[patient_id] = indices                # Add this patient's data
        # with open(self.log_txt, 'w') as f :           # Open JSON and write
        #     json.dump(old_dict, f)                    # new dictionary to file

        # TXT Implementation
        with open(self.log_txt, 'a') as f :
            new_line = "{}:{}\n".format(patient_id, str(indices))
            f.write(new_line)

        ### APPEND BINARY PREDICTION TO CSV ###
        with open(self.log_csv, 'a') as f :
            label = 1 if len(indices) > 0 else 0
            new_line = "{},{}\n".format(patient_id, str(label))
            f.write(new_line)

    def remove_body(self, image, sig=10, t=0.01) :
        try :
            otsu = threshold_otsu(image)                  # Compute Otsu threshold
        except ValueError :
            print("ALL the pixels are the same number?!")
            print(image)
            return None
        fill = bf(np.array(image > otsu, dtype=int))  # Fill holes
        gauss_fill = gaussian(fill, sigma=sig)        # Add Gaussian  blur
        fill = np.array(gauss_fill < t, dtype=int)    # Threshold again
        cropped = np.multiply(image, fill)            # Crop out body from raw image
        return cropped

    def detect_peaks(self, x) :
        ''' Detects peaks in a 1D array x.'''

        m = np.median(x)
        std = np.std(x)
        h = m + 4.*std

        peak_indices, _ = find_peaks(x,
                              distance=None,
                              threshold=None,
                              height=h,
                              prominence=4)
        return peak_indices

    def classify(self) :
        nb_patients = self.data_loader.dataset_length

        # Dictionary containing each patient and the slice indeces of the artifacts
        art_slices = {}
        class_list = [] # A list containing binary classifications (1=has artifact), same order as patient_list
        ''' art_slices has the form:
        {"patient1_ID": [slice_index1, slice_index2], # Patient1 has 2 slices with artifacts
         "patient2_ID": None,                         # Patient2 has no artifacts
         "patient3_ID": [slice_index1]}               # Patient3 has 1 slice with artifacts'''

        # Loop through all patients in the dataset (if not in test mode)
        iterations = 3 if self.test_mode==True else nb_patients

        # for i in range(nb_patients) :
        for i in range(iterations) : # Do only 4 patients for now
            pid = self.data_loader.patient_list[i]

            # Get the image Data
            stack, label = self.data_loader.getitem(i)
            stack = stack[ :, 0:350, :] # Limit the image range
                                            # This removes unwanted common features

            intensities = []

            # Loop through all images in patient's stack of scans
            for image in stack :
                if np.sum(image) < 1.0e-5 :
                    continue   # If the image is entirely black, just go to next image

                # Remove the patient's body from the images
                image = self.remove_body(image, sig=self.sigma, t=self.t1)

                # Threshold the new image
                image = np.array(image > self.t2, dtype=int)

                # Get sinogram
                theta = np.linspace(0., 180., 180, endpoint=False)
                sinogram = radon(image, theta=theta, circle=True)

                # Calculate mean intensity in key region of sinogram
                mean = np.mean(sinogram[100:-100, 40:-40])

                # Append to list of intensities
                intensities.append(mean)

            # Find the slices with artifacts
            print(max(intensities))
            indices = self.detect_peaks(intensities)


            # Add these to the classification Dictionary
            art_slices[pid] = indices

            # Write the predictions for this patient to memory
            self.save_patient(pid, indices)

        return art_slices


if __name__ == '__main__' :

    args, unparsed = get_args()

    dl = DataLoader(args)

    classifier = Classifier(args, dl)


    # Setup Parallel computing

    slices_dict = classifier.classify()
