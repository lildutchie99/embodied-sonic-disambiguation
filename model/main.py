"""
main.py

WN 2022 EECS 692 Final Project
Sound-reasoning

Sean Anderson, Nikhil Devraj, Ayush Shrivatsava,
and Ben VanDerPloeg (alphabetical by last name)
4/19/22
"""

import argparse
import os
import logging
from typing import Optional, Union, Dict, List, Tuple
import shutil

import numpy as np
# FIXME: MUST import torch AFTER import numpy... somehow my env got messed up
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
import librosa
from skimage.measure import block_reduce
from gym import spaces
import baseline as bl


RIR_SAMPLING_RATE = 16000  # seems to be what they always used for matterport3d


def compute_spectrogram(audio_data):
    # from sound-spaces/soundspaces/tasks/nav.py:SpectrogramSensor
    """My comment:
    inputs: audio_data is np.ndarray of shape (2, SAMPLINGRATE) where SAMPLINGRATE = 16000 for MP3D
            i.e. two audio waveforms, one for each ear
    return spectrogram which should be ndarray of shape (65, 26, 2) (one spectrogram for each ear)
    """

    def compute_stft(signal):
        n_fft = 512
        hop_length = 160
        win_length = 400
        stft = np.abs(
            librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
        stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
        return stft
    channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
    channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
    spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)
    return spectrogram


def gen_fake_trial() -> Dict[str, torch.Tensor]:
    """
    trial is batch size of 1
    """
    # define some fake data
    # TODO: min and max values for waveform
    WAV_MIN = -100
    WAV_MAX = 100
    RIR_SAMPLING_RATE = 16000  # for mp3d
    RGB_MIN = 0
    RGB_MAX = 256
    TRAJ_LEN = 8
    SOUND_TIMESTEP = 3

    rng = np.random.default_rng()
    gpss, spectrograms, rgbs = list(), list(), list()
    for i in range(TRAJ_LEN):
        if i == SOUND_TIMESTEP:
            binaural_wav = rng.uniform(WAV_MIN, WAV_MAX, size=(2, RIR_SAMPLING_RATE))
        else:
            binaural_wav = np.zeros((2, RIR_SAMPLING_RATE))
        spectrogram = torch.tensor(compute_spectrogram(binaural_wav), dtype=torch.float)
        image = torch.tensor(rng.integers(RGB_MIN, RGB_MAX, size=(128, 128, 3)), dtype=torch.float)
        gps = torch.tensor(rng.uniform(-10, 10, size=2), dtype=torch.float)
        rgbs.append(image)
        spectrograms.append(spectrogram)
        gpss.append(gps)

    # batch of size 1
    traj = {'spectrogram': torch.unsqueeze(torch.stack(spectrograms), 0),
            'rgb': torch.unsqueeze(torch.stack(rgbs), 0),
            'gps': torch.unsqueeze(torch.stack(gpss), 0)}

    return traj


def make_fake_batch(batch_size: int) -> Dict[str, torch.Tensor]:
    """
    make a batch larger than size 1
    MODIFIES: traj
    """
    trajs = [ gen_fake_trial() for _ in range(batch_size) ]
    # now need to put them in same batch
    traj_spects, traj_images, traj_coords = list(), list(), list()
    for traj in trajs:
        traj_spects.append(traj['spectrogram'])
        traj_images.append(traj['rgb'])
        traj_coords.append(traj['gps'])

    batch = {'spectrogram': torch.cat(traj_spects, 0),
             'rgb': torch.cat(traj_images, 0),
             'gps': torch.cat(traj_coords, 0)}
    return batch


def get_batch(dataX: Dict[str, torch.Tensor], batch_start: int,
              batch_end: int) -> Dict[str, torch.Tensor]:
    # requires:
    # get just examples from batch start to batch end
    s = dataX['spectrogram'][batch_start:batch_end]
    i = dataX['rgb'][batch_start:batch_end]
    g = dataX['gps'][batch_start:batch_end]

    return {'spectrogram': s, 'rgb': i, 'gps': g}


def fake_train_loop(model: bl.Tier1BaselineNet,
                    data: Union[DataLoader, Tuple[Dict[str, torch.Tensor]], torch.Tensor],
                    loss_fn, optimizer, batch_size: int) -> None:
    """
    REQUIRES:
    MODIFIES: model
    EFFECTS:  draft training loop for model
    """
    #optimizer = torch.optim.Adam(model.parameters())
    # example training
    dataX, datay= data[0], data[1]
    num_trajs = dataX['spectrogram'].shape[0]
    if num_trajs % batch_size != 0:
        raise NotImplementedError(f'just pass multiple of 10')
    for batch_start_idx in range(0, num_trajs, batch_size):
        # get batch starting at this index
        X = get_batch(dataX, batch_start_idx, batch_start_idx+batch_size)
        y = datay[batch_start_idx:batch_start_idx+batch_size]

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch_start_idx
        print(f"loss: {loss:>7f}  [{current}/{num_trajs}]")

    return


def fake_test_loop(model: bl.Tier1BaselineNet, data, loss_fn, batch_size: int):
    """

    Args:
        model:
        data:
        loss_fn:

    Returns:

    """
    correct, test_loss = 0, 0
    with torch.no_grad():
        if data is DataLoader:
            # closely following https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            raise NotImplementedError
        else:
            # example training
            dataX, datay = data[0], data[1]
            num_trajs = dataX['spectrogram'].shape[0]
            for i in range(0, num_trajs, batch_size):
                # get batch starting at this index
                X = get_batch(dataX, i, i+batch_size)
                y = datay[i:i + batch_size]

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= (num_trajs / batch_size)
            correct /= num_trajs
            print(
                f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def fake_train(model: bl.Tier1BaselineNet, data, testdata, batch_size: int):
    """
    MODIFIES: model
    Args:
        model:
        data:

    Returns:

    """
    # for classification
    loss_fn = nn.CrossEntropyLoss()
    # soundspaces uses Adam defaults, see page 9
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------")
        fake_train_loop(model, data, loss_fn, optimizer, batch_size)
        fake_test_loop(model, testdata, loss_fn, batch_size)


def init_model(ablate_sound: bool = False, ablate_vision: bool = False, ablate_gps: bool = False):
    # nuff said

    # just hard coded observation space
    SPECTROGRAM_SHAPE = (65, 26, 2)
    RGB_SHAPE = (128, 128, 3)
    spectrogram_obs_space = spaces.Box(low=np.finfo(np.float32).min,
                                       high=np.finfo(np.float32).max,
                                       shape=SPECTROGRAM_SHAPE,
                                       dtype=np.float32)
    visual_obs_space = {'rgb': spaces.Box(low=np.finfo(np.float32).min,
                                          high=np.finfo(np.float32).max,
                                          shape=RGB_SHAPE,
                                          dtype=np.float32)}
    gps_obs_space = spaces.Box(low=-11, high=11, shape=(3,), dtype=np.float32)
    obs_space = {'SpectrogramSensor': spectrogram_obs_space,
                 'RBGSensor': visual_obs_space,
                 'GPSSensor': gps_obs_space}

    HIDDEN_SIZE = 512  # see soundspaces paper page 9
    bn = bl.Tier1BaselineNet(obs_space, HIDDEN_SIZE, 15, ablate_sound, ablate_vision, ablate_gps)

    return bn


def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    bn = init_model(ablate_vision=False, ablate_sound=False, ablate_gps=True)

    # make fake dataset
    DATASET_SIZE = 100
    BATCH_SIZE = 10
    rng = np.random.default_rng()
    dataX = make_fake_batch(DATASET_SIZE)
    dataY = torch.tensor(rng.integers(low=0, high=2, size=DATASET_SIZE))
    data = (dataX, dataY)

    # try training, with validation being first batch... lol
    fakeval = (get_batch(data[0], 0, 2*BATCH_SIZE), data[1][:2*BATCH_SIZE])
    fake_train(bn, data, fakeval, batch_size=BATCH_SIZE)

    ## TEST: forward pass on single batch of trajectories
    #fakebatch = make_fake_batch(10)
    #ans = bn(fakebatch)
    #ans = functional.log_softmax(ans, dim=1)
    #print(ans)

    return 0


if __name__ == '__main__':
    main()
