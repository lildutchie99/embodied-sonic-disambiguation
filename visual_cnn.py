# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


# from sound-scapes/ss_baselines/common/utils.py
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def conv_output_dim(dimension, padding, dilation, kernel_size, stride
):
    r"""Calculates the output height and width based on the input
    height and width to the convolution layer.

    ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    assert len(dimension) == 2
    out_dimension = []
    for i in range(len(dimension)):
        out_dimension.append(
            int(
                np.floor(
                    (
                            (
                                    dimension[i]
                                    + 2 * padding[i]
                                    - dilation[i] * (kernel_size[i] - 1)
                                    - 1
                            )
                            / stride[i]
                    )
                    + 1
                )
            )
        )
    return tuple(out_dimension)


def layer_init(cnn):
    for layer in cnn:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                layer.weight, nn.init.calculate_gain("relu")
            )
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)


class VisualCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth components

    Args:
        sensor_space: The observation_space of the visual sensor (Dict)
                           map "rgb" to gym.spaces.Box?
                           map "depth" to gym.spaces.Box (or don't include this key)
        output_size: The size of the embedding vector
    """

    def __init__(self, sensor_space: Dict, output_size):
        super().__init__()
        if "rgb" in sensor_space:
            self._n_input_rgb = sensor_space["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in sensor_space:
            self._n_input_depth = sensor_space["depth"].shape[2]
        else:
            self._n_input_depth = 0

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (2, 2)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                sensor_space["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                sensor_space["depth"].shape[:2], dtype=np.float32
            )
        else:
            raise ValueError(f'how else to define cnn dimensions?')

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        layer_init(self.cnn)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        # permute tensor to dimension [BATCH x TIME x CHANNEL x HEIGHT X WIDTH]
        rgb_observations = observations.permute(0, 1, 4, 2, 3)
        rgb_observations = rgb_observations / 255.0  # normalize RGB
        shape = rgb_observations.shape
        # reshape to dimension [BATCH * TIME x CHANNEL x HEIGHT x WIDTH]
        flattened = rgb_observations.reshape(shape[0]*shape[1], shape[2], shape[3], shape[4])
        output = self.cnn(flattened)
        #if self._n_input_depth > 0:
        #    depth_observations = observations["depth"]
        #    # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        #    depth_observations = depth_observations.permute(0, 3, 1, 2)
        #    cnn_input.append(depth_observations)
        return output.reshape(shape[0], shape[1], -1)
