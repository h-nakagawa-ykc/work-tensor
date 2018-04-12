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

"""Central location for storage of useful constants."""

import tensorflow as tf
from tensorflow.python.client import device_lib

if tf.test.is_built_with_cuda():
  local_device_protos = device_lib.list_local_devices()
  NUM_GPUS = sum([1 for d in local_device_protos if d.device_type == "GPU"])
else:
  NUM_GPUS = 0
