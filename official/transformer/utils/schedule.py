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
"""Abstract training on a step or epoch basis."""

import tensorflow as tf

from official.transformer.utils import dataset


class Manager(object):
  """Container for convenience functions to abstract step or epoch basis.
  Transformer allows users to specify an epoch basis (generally recommended for
  full training) or a number of steps basis (convenient since epochs are rather
  large). TPUs furthermore require a step basis; however epochs are the norm in
  the machine learning community and it is desirable to allow users to specify
  epochs even when running with TPUS which requires behind the scenes
  conversions.
  This container simply groups what are largely mundane checks and conversions
  rather than interspersing them throughout the run loop code.
  """

  def __init__(self, train_steps, steps_between_evals, train_epochs,
      epochs_between_evals, default_train_epochs, batch_size,
      use_tpu=False, num_tpu_shards=8):
    if train_steps and train_epochs:
      raise ValueError("Both train_steps or train_epochs were be defined.")

    # Determine training schedule based on flags.
    if train_steps:
      self.train_eval_iterations = train_steps // steps_between_evals
      self._single_iteration_train_steps = steps_between_evals
      self._single_iteration_train_epochs = None
    else:
      train_epochs = train_epochs or default_train_epochs
      self.train_eval_iterations = train_epochs // epochs_between_evals
      self._single_iteration_train_steps = None
      self._single_iteration_train_epochs = epochs_between_evals

    self.batch_size = batch_size
    self.eval_batch_size = batch_size
    self.use_tpu = use_tpu

    if self.use_tpu and self.single_iteration_eval_steps == 0:
      num_eval = dataset.NUM_EXAMPLES[tf.estimator.ModeKeys.EVAL]
      new_eval_batch_size = num_eval // num_tpu_shards * num_tpu_shards
      tf.logging.info(
          "Evaluation batch size of {} is greater than the total number of "
          "evaluation examples ({}). TPU estimator does not permit partial "
          "batches; the evaluation batch size has been set to {}".format(
              self.eval_batch_size,
              num_eval,
              new_eval_batch_size
          ))
      self.eval_batch_size = new_eval_batch_size
      assert self.single_iteration_eval_steps == 1

    if self.use_tpu:
      assert self.batch_size % num_tpu_shards == 0
      assert self.eval_batch_size % num_tpu_shards == 0


  @property
  def single_iteration_train_steps(self):
    if self._single_iteration_train_steps or not self.use_tpu:
      return self._single_iteration_train_steps

    return dataset.epochs_to_steps(
        num_epochs=self._single_iteration_train_epochs,
        batch_size=self.batch_size,
        mode=tf.estimator.ModeKeys.TRAIN)

  @property
  def single_iteration_eval_steps(self):
    if not self.use_tpu:
      return None

    return dataset.epochs_to_steps(
        num_epochs=1,
        batch_size=self.eval_batch_size,
        mode=tf.estimator.ModeKeys.EVAL)

  @property
  def train_increment_str(self):
    if self._single_iteration_train_steps:
      return "{} steps.".format(self._single_iteration_train_steps)

    if not self.use_tpu:
      return "{} epochs.".format(self._single_iteration_train_epochs)

    return "~{} epochs. ({} steps)".format(
        self._single_iteration_train_epochs,
        self.single_iteration_train_steps)

  @property
  def repeat_dataset(self):
    # TODO(robieta@): handle TPU case of steps > 1 epoch
    return self._single_iteration_train_epochs
