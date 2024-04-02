import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import remotezip as rz

import tensorflow as tf

import time

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

start_time = time.time()

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'


def list_files_from_zip_url(zip_url):
    """ List the files in each class of the dataset given a URL with the zip file.

      Args:
        zip_url: A URL from which the files can be extracted from.

      Returns:
        List of files in each of the classes.
    """
    files = []
    with rz.RemoteZip(zip_url) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files


files = list_files_from_zip_url(URL)
files = [f for f in files if f.endswith('.avi')]
files[:10]


def get_class(fname):
    """ Retrieve the name of the class given a filename.

       Args:
         fname: Name of the file in the UCF101 dataset.

       Returns:
         Class that the file belongs to.
     """

    return fname.split('_')[-3]  # splits file name into separate parts after "_"


# pick the 3rd last elements from that split group


def get_files_per_class(files):
    """ Retrieve the files that belong to each class.

        Args:
          files: List of files in the dataset.

        Returns:
          Dictionary of class names (key) and files (values).
      """
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class


NUM_CLASSSES = 10
FILES_P_CLASS = 50

files_for_class = get_files_per_class(files)
classes = list(files_for_class.keys())

print('Num Classses', len(classes))
print('Num of videos for class[0]', len(files_for_class[classes[0]]))


def select_subset_of_classes(files_for_class, classes, files_per_class):
    """ Create a dictionary with the class name and a subset of the files in that class.

       Args:
         files_for_class: Dictionary of class names (key) and files (values).
         classes: List of classes.
         files_per_class: Number of files per class of interest.

       Returns:
         Dictionary with class as key and list of specified number of video files in that class.
     """
    files_subset = dict()

    for class_name in classes:
        class_files = files_for_class[class_name]
        files_subset[class_name] = class_files[:files_per_class]

    return files_subset


files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSSES], FILES_P_CLASS)
f_sub = list(files_subset.keys())
print(f_sub)

def download_from_zip(zip_url, to_dir, file_names):
    """ Download the contents of the zip file from the zip URL.

       Args:
         zip_url: A URL with a zip file containing data.
         to_dir: A directory to download data to.
         file_names: Names of files to download.
     """

    with rz.RemoteZip(zip_url) as zip:
        for fn in tqdm.tqdm(file_names):
            # tqdm.tqdm(file_names) is used to create a progress bar for the iteration, which provides visual
            # feedback on the progress of the loop. This line starts a loop that iterates over each file name (fn) in
            # the file_names list.
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzippped_file = to_dir / class_name / fn  # the path to the unzipped file

            fn = pathlib.Path(fn).parts[-1]
            # pathlib.Path(fn).parts returns a tuple containing the individual components of the file path,
            # and [-1] retrieves the last component, which is the file name itself.
            output_file = to_dir / class_name / fn  # same as the unzipped file path
            unzippped_file.rename(output_file)  # to move/rename the file


end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minute = (elapsed_time // 60)
elapsed_seconds = (elapsed_time % 60)

print("Elapsed time is: ", elapsed_minute, "minute(s) and ", elapsed_seconds, "seconds")
