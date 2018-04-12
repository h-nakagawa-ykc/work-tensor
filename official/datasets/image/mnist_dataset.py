#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from official.utils.misc import file_helpers


IMAGE_SIZE = 28
NUM_CLASSES = 10

NUM_IMAGES = {
  'train': 60000,
  'validation': 10000,
}


def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
  """Validate that filename corresponds to images for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_images, unused
    rows = read32(f)
    cols = read32(f)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))
    if rows != IMAGE_SIZE or cols != IMAGE_SIZE:
      raise ValueError(
          'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
          (f.name, rows, cols))


def check_labels_file_header(filename):
  """Validate that filename corresponds to labels for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_items, unused
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))

def download(directory, filename):
  """Download (and unzip) a file from the MNIST dataset if not already done."""
  filepath = os.path.join(directory, filename)
  url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
  file_helpers.download_and_extract(dest_directory=directory, data_url=url)
  return filepath


def construct_dataset(directory, images_file, labels_file):
  """Download and parse MNIST dataset."""

  images_file = download(directory, images_file)
  labels_file = download(directory, labels_file)

  check_image_file_header(images_file)
  check_labels_file_header(labels_file)

  def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE ** 2])
    return image / 255.0

  def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)

  images = tf.data.FixedLengthRecordDataset(
      images_file, IMAGE_SIZE * IMAGE_SIZE, header_bytes=16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
  return tf.data.Dataset.zip((images, labels))


def train(directory):
  """tf.data.Dataset object for MNIST training data."""
  return construct_dataset(directory, 'train-images-idx3-ubyte',
                           'train-labels-idx1-ubyte')


def test(directory):
  """tf.data.Dataset object for MNIST test data."""
  return construct_dataset(directory, 't10k-images-idx3-ubyte',
                           't10k-labels-idx1-ubyte')


def make_training_input_fn(data_dir, use_tpu, default_batch_size,
                           repeat_epochs=None):
  """Constructs an input function for Estimator to use during training.

  Args:
    data_dir: The location of the raw data files.
    use_tpu: Whether or not training will occur on a TPU.
    default_batch_size: The number of examples in each batch.
    repeat_epochs: The number of epochs that the dataset should yield. This
      parameter is forced to None if use_tpu is True."""
  # type: (str, bool, bool) -> function
  buffer_size = NUM_IMAGES["train"]
  repeat_epochs = None if use_tpu else repeat_epochs
  def input_fn(params):
    # type: (dict) -> tf.data.Dataset

    # The reason for this redundancy is a difference in the Estimator and
    # TPUEstimator APIs. TPUEstimator automatically populates batch_size, and
    # does not allow users to set that field. This code is compatible with
    # both APIs, which still respecting TPUEstimator's parameter.
    batch_size = params.get("batch_size", default_batch_size)

    dataset = train(data_dir).cache().repeat(repeat_epochs).shuffle(
        buffer_size=buffer_size)
    dataset = (
      # TPUs do not allow fractional batches.
      dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
      if use_tpu else dataset.batch(batch_size)
    )
    return dataset
  return input_fn


def make_eval_input_fn(data_dir, use_tpu, default_batch_size):
  """Constructs an input function for Estimator to use during evaluation.

  Args:
    data_dir: The location of the raw data files.
    use_tpu: Whether or not training will occur on a TPU.
    default_batch_size: The number of examples in each batch."""
  # type: (str, bool) -> function
  def input_fn(params):
    # type: (dict) -> tf.data.Dataset

    # See above.
    batch_size = params.get("batch_size", default_batch_size)

    dataset = test(data_dir)
    return (
      # TPUs do not allow fractional batches.
      dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
      if use_tpu else dataset.batch(batch_size)
    ).make_one_shot_iterator().get_next()
  return input_fn
