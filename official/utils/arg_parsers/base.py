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

import argparse

import tensorflow as tf

from official.utils.misc import constants


# Map string to (TensorFlow dtype, default loss scale)
DTYPE_MAP = {
  "fp16": (tf.float16, 128),
  "fp32": (tf.float32, 1),
}


class Parser(argparse.ArgumentParser):
  def __init__(self, simple_help=True, parents=None):
    preemptive_parser = argparse.ArgumentParser(add_help=False)
    preemptive_parser.add_argument("--helpful", action="store_true",
                                help="Detailed list of arguments.")
    parents = [preemptive_parser] +(parents or [])
    super(Parser, self).__init__(add_help=True, parents=parents,
                                 allow_abbrev=False)
    self.allow_parse_known = False

  def parse_args(self, args=None, namespace=None):
    args = args or []
    if "--helpful" in args:
      self.__init__(simple_help=False)
      args.append("-h")

    self.allow_parse_known = True
    args = super(Parser, self).parse_args(args=args, namespace=namespace)
    self.allow_parse_known = False
    self.secondary_arg_parsing(args=args)
    return args

  def parse_known_args(self, args=None, namespace=None):
    if not self.allow_parse_known:
      raise SyntaxError("Do not call parse_known_args() directly.")
    return super(Parser, self).parse_known_args(args=args, namespace=namespace)

  def secondary_arg_parsing(self, args):
    self.parse_dtype_info(args)
    self.parse_max_gpus(args)
    self.gpu_checks(args)
    self.tpu_checks(args)

  @staticmethod
  def parse_dtype_info(flags):
    """Convert dtype string to tf dtype, and set loss_scale default as needed.

    Args:
      flags: namespace object returned by arg parser.

    Raises:
      ValueError: If an invalid dtype is provided.
    """
    if ("dtype" not in vars(flags) or
        flags.dtype in (i[0] for i in DTYPE_MAP.values())):
      return  # Make function safe without dtype flag and idempotent

    try:
      flags.dtype, default_loss_scale = DTYPE_MAP[flags.dtype]
    except KeyError:
      raise ValueError("Invalid dtype: {}".format(flags.dtype))

    flags.loss_scale = flags.loss_scale or default_loss_scale

  @staticmethod
  def parse_max_gpus(flags):
    if "num_gpus" in vars(flags) and flags.num_gpus == -1:
      flags.num_gpus = constants.NUM_GPUS

  @staticmethod
  def gpu_checks(flags):
    if "num_gpus" in vars(flags) and flags.num_gpus > constants.NUM_GPUS:
      raise ValueError("{} GPUs specified, but only {} detected".format(
          flags.num_gpus,
          constants.NUM_GPUS
      ))

    if ("num_gpus" in vars(flags) and "batch_size" in vars(flags) and
        flags.num_gpus > 1 and flags.batch_size % flags.num_gpus != 0):
      # For multi-gpu, batch-size must be a multiple of the number of GPUs.
      #
      # Note that this should eventually be handled by replicate_model_fn
      # directly. Multi-GPU support is currently experimental, however,
      # so doing the work here until that feature is in place.
      remainder = batch_size % num_gpus
      err = ('When running with multiple GPUs, batch size '
             'must be a multiple of the number of available GPUs. Found {} '
             'GPUs with a batch size of {}; try --batch_size={} instead.'
             ).format(num_gpus, batch_size, batch_size - remainder)
      raise ValueError(err)

  @staticmethod
  def tpu_checks(flags):
    if "tpu" not in vars(flags) or flags.tpu is None:
      return

    if "num_gpus" in vars(flags) and flags.num_gpus > 0:
      tf.logging.warning("TPU flag passed. Setting num_gpu to zero.")
      flags.num_gpus = 0  # TPU takes precedence over GPU

    for key in ["data_dir", "model_dir", "export_dir"]:
      if key not in vars(flags) or vars(flags)[key] is None:
        continue
      if not vars(flags)[key].startswith("gs://"):
        raise ValueError("Invalid value {} for flag '{}'. TPUs must be run "
                         "using GCS storage.".format(vars(flags)[key], key))
