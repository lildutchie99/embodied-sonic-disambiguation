# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from typing import Dict

from audio_cnn import AudioCNN, Flatten
from visual_cnn import VisualCNN, layer_init


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class RNNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        r"""An RNN for encoding the state in RL.

        Supports masking the hidden state during various timesteps in the forward loss

        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type
        # TODO: LSTM option?
        if rnn_type == "LSTM":
            raise DeprecationWarning(f" don't use lstm for now")

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )

        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )

        return hidden_states

    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states

        return hidden_states

    def single_forward(self, x, hidden_states, masks):
        r"""Forward for a non-sequence input
        """
        hidden_states = self._unpack_hidden(hidden_states)
        x, hidden_states = self.rnn(
            x.unsqueeze(0),
            self._mask_hidden(hidden_states, masks.unsqueeze(0)),
        )
        x = x.squeeze(0)
        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def seq_forward(self, x, hidden_states, masks):
        r"""Forward for a sequence of length T

        Args:
            x: (T, N, -1) Tensor that has been flattened to (T * N, -1)
            hidden_states: The starting hidden state.
            masks: The masks to be applied to hidden state at every timestep.
                A (T, N) tensor flatten to (T * N)
        """
        raise DeprecationWarning
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(x.size(0) / n)

        # unflatten
        x = x.view(t, n, x.size(1))
        masks = masks.view(t, n)

        # steps in sequence which have zero for any agent. Assume t=0 has
        # a zero in it.
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]  # handle scalar
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [t]

        hidden_states = self._unpack_hidden(hidden_states)
        outputs = []
        for i in range(len(has_zeros) - 1):
            # process steps that don't have any zeros in masks together
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, hidden_states = self.rnn(
                x[start_idx:end_idx],
                self._mask_hidden(
                    hidden_states, masks[start_idx].view(1, -1, 1)
                ),
            )

            outputs.append(rnn_scores)

        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)
        x = x.view(t * n, -1)  # flatten

        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states

    def forward(self, x, initial_hidden_states=None):
        # x should have dimension [BATCH x TIME x FEATURE]
        # hidden states assumed 0 by GRU default
        # hiddenstates should be of shape [BATCH x TIME x FEATURE]?
        # no need to permute, batch_first is true
        return self.rnn(x, initial_hidden_states)
        #return self.seq_forward(x, hidden_states, masks)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass


class Classifier(nn.Module):
    # a really simple feedforward classifier.
    def __init__(self, input_size: int, num_outputs: int):
        super(Classifier, self).__init__()
        HIDDEN_SIZES = [512, 128, 64]
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], HIDDEN_SIZES[2]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[2], num_outputs)
        )

        # TODO: better weight initialization?
        layer_init(self.net)

    def forward(self, x):
        # should be ok with [BATCH x FEATURE]
        return self.net(x)


class Tier1BaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, num_outputs: int,
                 ablate_sound: bool = False, ablate_vision: bool = False, ablate_gps: bool = False):
        # num_outputs: number of objects to classify
        # 4/26/22 meeting: only train on trajectories from single environment (house),
        #                  candidate objects are ALL objects from house (i.e. set of
        #                  trajectories trained on). Philosophy is that the classification
        #                  head would be fine-tuned for other environments but the underlying
        #                  model weights would stay the same.
        if ablate_sound and ablate_gps and ablate_vision:
            raise ValueError(f"don't ablate everything")
        super().__init__()
        self._hidden_size = hidden_size
        self.ablate_sound = ablate_sound
        self.ablate_vision = ablate_vision
        self.ablate_gps = ablate_gps

        if not ablate_vision:
            try:
                rgb_shape = observation_space['RBGSensor']['rgb'].shape
            except KeyError:
                raise ValueError(f'need to provide RGB observation space')
            self.visual_encoder = VisualCNN(observation_space['RBGSensor'], hidden_size)
            #print('rgb torchsummary')
            #summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]),
            #        device='cpu')
        if 'depth' in observation_space:
            raise NotImplementedError(f'depth is not included in model right now')
            depth_shape = observation_space['depth'].shape
            #summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]),
            #        device='cpu')

        if not ablate_sound:
            audiogoal_sensor = 'SpectrogramSensor'
            self.audio_encoder = AudioCNN(observation_space['SpectrogramSensor'], hidden_size,
                                          audiogoal_sensor)
            audio_shape = observation_space[audiogoal_sensor].shape
            #print('audio torchsummary')
            #summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]),
            #        device='cpu')

        # should be gps size + audio CNN output size + visual CNN output size, factoring in
        # ablations
        # (does not include hidden state)
        audio_output_size = self._hidden_size if not ablate_sound else 0
        vision_output_size = self._hidden_size if not ablate_vision else 0
        gps_output_size = observation_space['GPSSensor'].shape[0] if not ablate_gps else 0
        rnn_input_size = gps_output_size + audio_output_size + vision_output_size
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        # input is last hidden state of GRU
        self.classifier = Classifier(hidden_size, num_outputs)

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations: Dict[str, torch.Tensor], rnn_hidden_states: torch.Tensor = None,
                masks=None) -> torch.Tensor:
        # for ablations, just ignore that part of the input
        x = []
        if not self.ablate_sound:
            x.append(self.audio_encoder(observations['spectrogram']))
        if not self.ablate_vision:
            x.append(self.visual_encoder(observations['rgb']))
        if not self.ablate_gps:
            x.append(observations['gps'])

        # first two dimensions are BATCH, TIME
        x1 = torch.cat(x, dim=2)
        x2, last_hidden_state = self.state_encoder(x1, rnn_hidden_states)

        assert not torch.isnan(x2).any().item()

        # run feedforward classifier on hidden state, flatten everything except batch dim
        y = self.classifier(torch.squeeze(last_hidden_state))
        return y
