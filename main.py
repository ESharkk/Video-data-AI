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

from tensorflow.python.data import AUTOTUNE
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
            unzipped_file = to_dir / class_name / fn  # the path to the unzipped file

            fn = pathlib.Path(fn).parts[-1]
            # pathlib.Path(fn).parts returns a tuple containing the individual components of the file path,
            # and [-1] retrieves the last component, which is the file name itself.
            output_file = to_dir / class_name / fn  # same as the unzipped file path
            if not output_file.exists():
                unzipped_file.rename(output_file)  # Rename the file
            else:
                os.remove(unzipped_file)  # to move/rename the file


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
            files.remove(f)  # removes the file from the list if it does not have a name
        files_for_class = get_files_per_class(files)

        classes = list(files_for_class.keys())[:num_classes]

        for cls in classes:
            random.shuffle(files_for_class)

        # only ues the number of classes you want in the dictionary
        files_for_class = {x: files_for_class[x] for x in classes}

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

video_count_train = len(list(download_dir.glob('train/*/*.avi')))
video_count_test = len(list(download_dir.glob('test/*/*.avi')))
video_count_val = len(list(download_dir.glob('val/*/*.avi')))
video_total = video_count_test + video_count_train + video_count_val
print(f"Total videos : {video_total}")


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame : Image that needs to be resized and padded.
        output_size: Pixel size of the output frame image.

    Return
        Formatted frame with padding of specified output_size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    """
    Creates frames from the video files.

    Args:
         video_path: File path to the video.
         n_frames: Number of frames to be produced per video.
         output_size: Pixel size of the output video image.
    Return:
        An numpy array of frames in shape of (n_frames, height, width, channels)
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    needed_length = 1 + (n_frames - 1) * frame_step

    if needed_length > video_length:
        start = 0
    else:
        max_start = video_length - needed_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret os a boolean indicating whether read was successful, frame is the image itself

    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))
    src.release()
    result = np.array(result)

    return result


video_path = "End_of_a_jam.ogv"

sample_video = frames_from_video_file(video_path, n_frames=10)
sample_video.shape


def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=10)
    return embed.embed_file('./animation.gif')


to_gif(sample_video)

# docs-infra: no-execute

ucf_sample_video = frames_from_video_file(next(subset_paths['train'].glob('*/*.avi')), 50)
to_gif(ucf_sample_video)


class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        """ Returns a set of frames with their associated label.

              Args:
                path: Video file paths.
                n_frames: Number of frames.
                training: Boolean to determine if training dataset is being created.
            """
        self.path = path
        self.n_frames = n_frames
        self.training = training
        # creates a list of class names based on the directory structure
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        # assigns unique ids tp each class name and stores them in a dictionary
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

        # extracts video file paths and their class names
    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

        # generates the frames
    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name]  # encode labels
            yield video_frames, label


# testing out the frame generator
fg = FrameGenerator(subset_paths['train'], 10, training=True)

frames, label = next(fg())

print(f"shape: {frames.shape}")
print(f"Label: {label}")

# create a training set

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], 10, training=True),
                                          output_signature=output_signature)

for frames, labels in train_ds.take(10):
    print(labels)

# create a validation set

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], 10),
                                        output_signature=output_signature)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)

# video training is done with a five dimensional object containing: [batch_size, num_of_frames, height, width, channels]
# whereas the image training is done with four dimension: [batch_size, height, width, channels]

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames is: {train_frames.shape}')
print(f'Shape of training labels is: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames is: {val_frames.shape}')
print(f'Shape of validation labels is: {val_labels.shape}')

net = tf.keras.applications.EfficientNetB0(include_top=False)
net.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=255),
    tf.keras.layers.TimeDistributed(net),
    tf.keras.layers.Dense(10),
    tf.keras.layers.GlobalAveragePooling3D()
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy'])

model.fit(train_ds,
          epochs=10,
          validation_data=val_ds,
          callbacks= tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')
          )

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minute = (elapsed_time // 60)
elapsed_seconds = (elapsed_time % 60)

print("Elapsed time is: ", elapsed_minute, "minute(s) and ", elapsed_seconds, "seconds")
