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

    return fname.split('_')[-3] # splits file name into separate parts after "_"
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

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minute = (elapsed_time // 60)
elapsed_seconds = (elapsed_time % 60)

print("Elapsed time is: ", elapsed_minute, "minute(s) and ", elapsed_seconds, "seconds")
