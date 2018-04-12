# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Parser for running on hardware accelerators."""

import argparse

import tensorflow as tf


class Parser(argparse.ArgumentParser):
  def __init__(self, add_help=False, simple_help=True, num_gpus=True, tpu=True):
    super(Parser, self).__init__(add_help=add_help)

    if num_gpus:
      self.add_argument(
          "--num_gpus", "-ng",
          type=int,
          default=1 if tf.test.is_built_with_cuda() else 0,
          help="[default: %(default)s] How many GPUs to use with the "
               "DistributionStrategies API. The default is 1 if TensorFlow was"
               "built with CUDA, and 0 otherwise.",
          metavar="<NG>"
      )

    if tpu:
      self.add_argument(
          "--tpu", "-t",
          help="[default: %(default)s] The Cloud TPU to use for training. This "
               "should be either the name used when creating the Cloud TPU, or "
               "a grpc://ip.address.of.tpu:8470 url.",
          metavar="<TPU>"
      )

      self.add_argument(
          "--tpu_zone", "-tz",
          help=(argparse.SUPPRESS if simple_help else
                "[Optional] GCE zone where the Cloud TPU is located in. If not "
                "specified, we will attempt to automatically detect the GCE "
                "project from metadata."),
          metavar="<TZ>"
      )

      self.add_argument(
          "--tpu_gcp_project", "-tgp", default=None,
          help=(argparse.SUPPRESS if simple_help else
                "[Optional] Project name for the Cloud TPU-enabled project. If "
                "not specified, we will attempt to automatically detect the "
                "GCE project from metadata."),
          metavar="<TGP>"
      )

      self.add_argument(
          "--num_tpu_shards", "-nts", type=int, default=8,
          help=(argparse.SUPPRESS if simple_help else
                "[default: %(default)s] Number of shards (TPU chips)."),
          metavar="<NTS>"
      )

      # TODO(robieta@): Determine if this can be a derived quantity.
      self.add_argument(
          "--iterations_per_loop", "-ipl", type=int, default=50,
          help=(argparse.SUPPRESS if simple_help else
                "[default: %(default)s] Iterations per TPU training loop."),
          metavar="<IPL>"
      )
