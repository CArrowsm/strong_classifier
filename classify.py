import os
import numpy as np
import pandas as pd
from data_loader import DataLoader

from scipy.ndimage.morphology import binary_fill_holes as bf
from skimage.filters import threshold_otsu, gaussian, apply_hysteresis_threshold

from skimage.transform import probabilistic_hough_line, radon, iradon, rescale
from skimage.feature import canny

'''
This script classifies images in the RadCure dataset which contain 'strong'
metal artifact streaks.

The program processes each patient slice-by-slice, and picks out slices with
where there is very strong evidence of artifacts.
'''

class Classifier(object):
    """docstring for Classifier."""
    def __init__(self, arg):
        # super(Classifier, self).__init__()
        self.data_loader = DataLoader(args)

        self.sigma = 10 # Width of Gaussian for blur
        self.t1 = 0.01  # Threshold for removal of body
        self.t2 = 0.01  # Threshold after body is removed


    def remove_body(self, image, sig=10, t=0.01) :
        otsu = threshold_otsu(image)                  # Compute Otsu threshold
        fill = bf(np.array(image > otsu, dtype=int))  # Fill holes
        gauss_fill = gaussian(fill, sigma=sig)        # Add Gaussian  blur
        fill = np.array(gauss_fill < t, dtype=int)    # Threshold again
        cropped = np.multiply(image, fill)            # Crop out body from raw image
        return cropped

    def detect_peak(self, x) :
        ''' Detects peaks in a 1D array x.'''
        pass

    def classify(self) :
        # patient_ids = self.data_loader.patient_list
        nb_patients = self.data_loader.dataset_length

        # Loop through all patients in the dataset
        for i in range(nb_patients) :
            pid = self.data_loader.patient_list[i]

            # Get the image Data
            image = self.data_loader.__getitem__(i)

            # 



if __name__ == '__main__' :

    args, unparsed = get_args()

    # dl = DataLoader(args)
