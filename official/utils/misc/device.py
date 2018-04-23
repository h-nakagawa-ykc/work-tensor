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
"""Collection of methods to handle corner cases for various device oddities."""

import functools

import tensorflow as tf

def retry_timeouts(num_attempts=1):
  """Retries functions which raise DeadlineExceededError.

  Args:
    num_attempts: The number of times a function will be attempted before
      raising."""

  def wrapper(f):
    @functools.wraps(f)
    def decorated_fn(*args, **kwargs):
      for i in range(num_attempts):
        try:
          return f(*args, **kwargs)
        except tf.errors.DeadlineExceededError:
          if i < (num_attempts - 1):
            tf.logging.error("DeadlineExceededError captured.")
          else:
            tf.logging.error("Max attempts reached. Raising.")
            raise
    return decorated_fn
  return wrapper


def construct_estimator(flags, use_tpu, model_fn, params):
  """Construct an estimator for a variety of devices.

  Args:
    flags:  The flags object from argument parsing.
    use_tpu:  A boolean indicating whether a TPU is to be used.
    model_fn: The model function to be passed to the estimator.
    params: A dict to be passed to the estimator.
  """
  session_config=tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=use_tpu
  )

  if use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=flags.tpu,
        zone=flags.tpu_zone,
        project=flags.tpu_gcp_project
    )
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=flags.model_dir,
        session_config=session_config,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=flags.iterations_per_loop,
            num_shards=flags.num_tpu_shards)
    )
    return tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=True,
        train_batch_size=flags.batch_size,
        eval_batch_size=flags.batch_size,
        params=params,
        config=run_config)

  if flags.num_gpus == 0:
    distribution = tf.contrib.distribute.OneDeviceStrategy('device:CPU:0')
  elif flags.num_gpus == 1:
    distribution = tf.contrib.distribute.OneDeviceStrategy('device:GPU:0')
  else:
    distribution = tf.contrib.distribute.MirroredStrategy(
        num_gpus=flags.num_gpus
    )

  run_config = tf.estimator.RunConfig(
      model_dir=flags.model_dir,
      train_distribute=distribution,
      session_config=session_config
  )
  return tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params=params
  )
