#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.datasets.image import mnist_dataset
from official.utils.arg_parsers import accelerator
from official.utils.arg_parsers import base
from official.utils.arg_parsers import parsers
from official.utils.logs import hooks_helper
from official.utils.misc import device


from official.utils.misc import model_helpers

LEARNING_RATE = 1e-4


def create_model(data_format, image_size=mnist_dataset.IMAGE_SIZE):
  # type: (str, int) -> tf.keras.Sequential
  """Model to recognize digits in the MNIST dataset.

  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

  But uses the tf.keras API.

  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
    image_size: The side length of an image. Images are assumed to be
      image_size x image_size square images. MNIST images are 28x28 pixels.

  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, image_size, image_size]
  else:
    assert data_format == 'channels_last'
    input_shape = [image_size, image_size, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Reshape(input_shape),
          l.Conv2D(
              filters=32,
              kernel_size=5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              filters=64,
              kernel_size=5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])


def construct_host_call(loss, labels, logits):
  """Create a training host call.

  https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec

  During TPU training, summary ops need to be executed on the CPU rather than
  the TPU. To this end TPUEstimatorSpec provides a host_call field to specify
  ops to be run on the CPU. There are, however, several significant
  restrictions:
    1) The host call function must be purely functional. Do not use ops which
       have already been created when defining the host call, as XLA rewrites
       ops during TPU training.
    2) Inputs must be Tensors with shape [batch]. (Hence the tf.reshape())
  """
  def host_call_fn(lr, ls, ac):
    return [
      tf.identity(lr[0], "learning_rate"),
      tf.identity(ls[0], "cross_entropy"),
      tf.identity(ac[0], "train_accuracy"),
      tf.summary.scalar("train_accuracy", ac[0])
    ]
  lr_t = tf.reshape(tf.Variable(LEARNING_RATE, dtype=tf.float32), [1])
  ls_t = tf.reshape(loss, [1])
  ac_t = tf.reshape(device.accuracy(labels=labels, logits=logits)[1], [1])
  return host_call_fn, [lr_t, ls_t, ac_t]


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  use_tpu = params["use_tpu"]  # type: bool

  model = create_model(params['data_format'])  # type: tf.keras.Sequential
  image = features
  if isinstance(image, dict):
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    if use_tpu:
      raise RuntimeError("mode PREDICT is not yet supported for TPUs.")

    logits = model(image, training=False)
    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
          'classify': tf.estimator.export.PredictOutput(predictions)
        })

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    if use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    logits = model(image, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    host_call = construct_host_call(loss=loss, labels=labels, logits=logits)

    minimize_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    spec_args = dict(
        mode=mode,
        loss=loss,
        train_op=tf.group(minimize_op, update_ops)
    )
    if use_tpu:
      # spec_args["host_call"] = host_call
      return tf.contrib.tpu.TPUEstimatorSpec(**spec_args)
    host_call[0](*host_call[1])
    return tf.estimator.EstimatorSpec(**spec_args)

  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels_in, logits_in):
      return {"accuracy": device.accuracy(labels=labels_in, logits=logits_in)}

    logits = model(image, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, [labels, logits])
      )
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=metric_fn(labels_in=labels, logits_in=logits)
    )


def construct_estimator(flags, use_tpu):
  use_gpu = flags.num_gpus > 0

  data_format = flags.data_format
  if data_format is None:
    data_format = ('channels_first' if use_gpu or use_tpu else 'channels_last')

  params = {"use_tpu": use_tpu, "data_format": data_format}
  return device.construct_estimator(flags=flags, use_tpu=use_tpu,
                                    model_fn=model_fn, params=params)


def main(argv):
  parser = MNISTArgParser()
  flags = parser.parse_args(args=argv[1:])
  use_tpu = flags.tpu is not None

  mnist_classifier = construct_estimator(flags=flags, use_tpu=use_tpu)

  # Set up hook that outputs training logs every 100 steps.
  train_hooks = hooks_helper.get_train_hooks(
      flags.hooks, batch_size=flags.batch_size)
  train_hooks = []

  if use_tpu:
    max_train_steps = (mnist_dataset.NUM_IMAGES["train"] *
                       flags.epochs_between_evals // flags.batch_size)
    eval_steps = mnist_dataset.NUM_IMAGES["validation"] // flags.batch_size
  else:
    max_train_steps, eval_steps = None, None

  # @device.retry_timeouts(num_attempts=1)
  def train():
    train_input_fn = mnist_dataset.make_training_input_fn(
        data_dir=flags.data_dir, use_tpu=use_tpu,
        default_batch_size=flags.batch_size,
        repeat_epochs=flags.epochs_between_evals)

    mnist_classifier.train(input_fn=train_input_fn, hooks=train_hooks,
                           steps=max_train_steps)

  # @device.retry_timeouts(num_attempts=1)
  def evaluate():
    eval_input_fn = mnist_dataset.make_eval_input_fn(
        data_dir=flags.data_dir, use_tpu=use_tpu,
        default_batch_size=flags.batch_size)

    return mnist_classifier.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)

  # Train and evaluate model.
  for _ in range(flags.train_epochs // flags.epochs_between_evals):
    train()
    eval_results = evaluate()
    print('\nEvaluation results:\n\t%s\n' % eval_results)

    if model_helpers.past_stop_threshold(
        flags.stop_threshold, eval_results['accuracy']):
      break

  # Export the model
  if flags.export_dir is not None:
    image = tf.placeholder(
        tf.float32, [None, mnist_dataset.NUM_IMAGES, mnist_dataset.NUM_IMAGES])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    mnist_classifier.export_savedmodel(flags.export_dir, input_fn)


class MNISTArgParser(base.Parser):
  """Argument parser for running MNIST model."""

  def __init__(self, simple_help=True):
    super(MNISTArgParser, self).__init__(parents=[
        parsers.BaseParser(multi_gpu=False),
        accelerator.Parser(simple_help=simple_help, num_gpus=True, tpu=True),
        parsers.ImageModelParser(),
        parsers.ExportParser()],
        simple_help=simple_help
    )

    self.set_defaults(
        data_dir='/tmp/mnist_data',
        model_dir='/tmp/mnist_model',
        batch_size=100,
        train_epochs=40)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
