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
