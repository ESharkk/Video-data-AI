

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


def split_class_lists(files_for_class, count):
    """ Returns the list of files that hasn't already been placed into a subset of data as well as the remainder of
       files that need to be downloaded.

       Args:
         files_for_class: Files belonging to a particular class of data.
         count: Number of files to download.

       Returns:
         Files belonging to the subset of data and dictionary of the remainder of files that need to be downloaded.
     """
    split_files = []
    remainder = {}
    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:count])
        remainder[cls] = files_for_class[cls][count:]
    return split_files, remainder


def download_ucf_101_subset(zip_url, num_classes, splits, download_dir):
    files = list_files_from_zip_url(zip_url)
    for f in files:
        path = os.path.normpath(f)
        tokens = path.split(os.sep)
        if len(tokens) <= 2:
            files.remove(f) # removes the file from the list if it does not have a name
        files_for_class = get_files_per_class(files)

        classes = list(files_for_class.keys())[:num_classes]

        for cls in classes:
            random.shuffle(files_for_class)

        # only ues the number of classes you want in the dictionary
        files_for_class ={x: files_for_class[x] for x in classes}

        dirs = {}
        for split_name, split_count, in splits.items():
            print(split_name, ":")
            split_dir = download_dir / split_name
            split_files, files_for_class = split_class_lists(files_for_class, split_count)
            download_from_zip(zip_url, split_dir, split_files)
            dirs[split_name] = split_dir

        return dirs

download_dir = pathlib.Path('/UCF101_subset')
subset_paths = download_ucf_101_subset(URL,
                                       num_classes=NUM_CLASSSES,
                                       splits={"train": 30, "val": 10, "test": 10},
                                       download_dir=download_dir)




end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minute = (elapsed_time // 60)
elapsed_seconds = (elapsed_time % 60)

print("Elapsed time is: ", elapsed_minute, "minute(s) and ", elapsed_seconds, "seconds")
