# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helper functions to automate downloading of small datasets.

The download helper uses tf.gfile where appropriate, and as a result allows
downloading to distributed filesystems."""

import functools
import gzip
import os
import re
import shutil
import sys
import tarfile
import tempfile

from six.moves import urllib

import tensorflow as tf


def download_and_extract(dest_directory, data_url):
  # type: (str, str) -> None
  filename = os.path.split(data_url)[1]
  filename = re.sub(r"(\.tar)?\.gz$", "", filename)
  if tf.gfile.Exists(os.path.join(dest_directory, filename)):
    tf.logging.info("{} already exists. {} will not be downloaded.".format(
        dest_directory, data_url
    ))
    return

  if data_url.endswith(".tar.gz"):
    _download_tarball(dest_directory=dest_directory, data_url=data_url)
  elif data_url.endswith(".gz"):
    _download_gzip(dest_directory=dest_directory, data_url=data_url)
  else:
    _download_uncompressed(dest_directory=dest_directory, data_url=data_url)


def _progress(count, block_size, total_size, filename):
  sys.stdout.write("\r>> Downloading %s %.1f%%" % (
    filename, min([100.0 * count * block_size / total_size, 100.0])))
  sys.stdout.flush()


def _download_tarball(dest_directory, data_url):
  download_dir = tempfile.mkdtemp()
  try:
    tarball_path = os.path.join(download_dir, "data.tar.gz")
    tarball_path, _ = urllib.request.urlretrieve(
        data_url, tarball_path,
        functools.partial(_progress, filename=tarball_path)
    )

    print()
    tarfile.open(tarball_path, 'r:gz').extractall(download_dir)
    os.remove(tarball_path)

    for root, _, fnames in os.walk(download_dir):
      dest_root = os.path.join(dest_directory, root[len(download_dir)+1:])
      tf.gfile.MakeDirs(dest_root)
      for fname in fnames:
        source_path = os.path.join(root, fname)
        dest_path = os.path.join(dest_root, fname)
        if dest_path.startswith("gs://"):
          print(">> Uploading {} to {}".format(source_path, dest_path))
        tf.gfile.Copy(source_path, dest_path)

  finally:
    shutil.rmtree(download_dir)


def _download_gzip(dest_directory, data_url):
  _, zipped_path = tempfile.mkstemp(suffix=".gz")
  try:
    unzipped_path = os.path.join(dest_directory, os.path.split(data_url)[1][:-3])
    zipped_path, _ = urllib.request.urlretrieve(
        data_url, zipped_path,
        functools.partial(_progress, filename=zipped_path)
    )
    print()

    tf.gfile.MakeDirs(dest_directory)
    with gzip.open(zipped_path, 'rb') as f_in, \
        tf.gfile.Open(unzipped_path, 'wb') as f_out:
      if unzipped_path.startswith("gs://"):
        print(">> Uploading {} to {}".format(zipped_path, unzipped_path))
      shutil.copyfileobj(f_in, f_out)
  finally:
    os.remove(zipped_path)


def _download_uncompressed(dest_directory, data_url):
  _, temp_path = tempfile.mkstemp()
  try:
    temp_path, _ = urllib.request.urlretrieve(
        data_url, temp_path,
        functools.partial(_progress, filename=temp_path)
    )
    dest_path = os.path.join(dest_directory, os.path.split(data_url)[1])
    tf.gfile.Copy(temp_path, dest_path)
  finally:
    os.remove(temp_path)
