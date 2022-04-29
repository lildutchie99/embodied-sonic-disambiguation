"""
trainer.py


main training code for networks.

EECS 692 WN22 Project
Sean Anderson
Nikhil Devraj
Ayush Shrivatsava
and Ben VanDerPloeg (alphabetical by last name)
April 2022
"""


# FIXME: MUST import torch AFTER import numpy... somehow my env got messed up
import numpy as np
import torch
from torchvision import io
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from scipy import signal
import json
from scipy.io import wavfile
from typing import Dict, Optional

import model.baseline as bl
from model.main import init_model, compute_spectrogram


RIR_SAMPLING_RATE = 16000  # for mp3d


class Tier1Dataset(Dataset):
    def __init__(self, config_file: str, img_dir: str, sound_dir: str,
                 max_traj_len: int, objidmapping_path: str = 'data/objid_to_index_17DRP5sb8fy.json'):
        """
        build dataset (should be a single split
        imgdir and sounddir can be same across splits. just config_file needs to be different
        Args:
            config_file: path to ben's config file
            img_dir: directory containing traj images, should contain folders named same as traj id
            sound_dir: directory contianing sounds wav
        """
        self.img_dir = img_dir
        self.sound_dir = sound_dir
        self.max_traj_len = max_traj_len

        with open(config_file, 'r') as f:
            self.dataset_config = json.load(f)

        self.index_mapping = {k: traj_id for k, traj_id in zip(range(len(self.dataset_config)),
                                                               self.dataset_config.keys())}
        with open(objidmapping_path, 'r') as f:
            self.obj_to_id_mapping = json.load(f)

        # can't batch tensors of different shapes
        # get max traj length

        self.BLANK_SPECTROGRAM = compute_spectrogram(np.zeros((2, RIR_SAMPLING_RATE)))

    def __len__(self):
        # assumes one trajectory is one scenario (datapoint)
        return len(self.dataset_config)

    def _pad_time(self, x: torch.tensor) -> torch.tensor:
        return functional.pad(x, (0, 0)*(len(x.shape)-1) + (0, self.max_traj_len - x.shape[0]))

    def __getitem__(self, idx):
        # REQUIRES: assuming idx 0-based
        #           num_gps == num rgb.png files >= timestep_to_play_sound >= 0

        traj = {}
        # assuming one scenario (datapoint) per trajectory, hence 0
        SCENARIO_IDX = 0  # will use if we want multiple scenarios per trajectory

        traj_id = self.index_mapping[idx]
        traj_config = self.dataset_config[traj_id][SCENARIO_IDX]
        traj_len = len(traj_config['gps'])

        # parse sound file
        #wav_path = self.sound_dir + '/' + self.dataset_config[traj_id][0]['causorsound']
        wav_path = self.sound_dir + f'/sound_{traj_id}_{SCENARIO_IDX}.wav'
        # TODO: resample? scipy.signal.resample
        samplerate, data = wavfile.read(wav_path)
        #if data.shape != (2, RIR_SAMPLING_RATE):
        #    print('weird:', wav_path, samplerate, data)
        if samplerate != RIR_SAMPLING_RATE:
            print('SHOULD NEVER BE HERE trainer.py:89')
            data = signal.resample(data, RIR_SAMPLING_RATE)
        if data.shape[0] != 2:
            data = data.T
            #raise NotImplementedError(f' change sampling rate from {samplerate} to'
            #                          f'{RIR_SAMPLING_RATE}')
        spectrogram = compute_spectrogram(data)

        # build each timestep of traj trial
        traj_img_dir = self.img_dir + '/' + str(traj_id) + '/'
        traj_imgs = []
        traj_spects = []
        for timestep in range(traj_len):
            # load image
            img_path = traj_img_dir + str(timestep) + '_rgb.png'
            img = io.read_image(img_path).permute(1, 2, 0)
            traj_imgs.append(img.type(torch.float))

            # load spectrogram
            if timestep == traj_config['timestep_to_play_sound']:
                this_spect = torch.tensor(spectrogram)
            else:
                this_spect = torch.tensor(self.BLANK_SPECTROGRAM)
            traj_spects.append(this_spect.type(torch.float))

        traj['rgb'] = self._pad_time(torch.stack(traj_imgs)).cuda()
        traj['spectrogram'] = self._pad_time(torch.stack(traj_spects)).cuda()
        traj['gps'] = self._pad_time(torch.tensor(traj_config['gps'])).cuda()

        # get gt label
        # FIXME: get contiguous object id label globally from json
        label = self.obj_to_id_mapping[str(traj_config['causorid'])]

        #https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
        # automatic batching should handle our Dict data structure!!
        return traj, label


def split_dataset(full_metadata: str, n: Optional[int] = None) -> None:
    """
    takes full metadata json and randomly splits into
    train, test, and val sets, so that all val and test ground truth
    labels appear in train set and split is about
    80/10/10
    Args:
        full_metadata:
        n:

    Returns:

    """
    rng = np.random.default_rng()
    with open(full_metadata, 'r') as f:
        metadata = json.load(f)

    # prune null scenarios
    metadata = {k: metadata[k] for k in metadata.keys() if len(metadata[k]) != 0}

    if n is not None:
        # only use first n trajs in metadata
        traj_idx2id = list(metadata.keys())[:n]
        # construct a smaller metadata
        metadata = { k: metadata[k] for k in traj_idx2id }

    traj_idx2id = list(metadata.keys())
    num_trials = len(metadata)
    test_size = num_trials // 10
    indices = rng.integers(low=0, high=num_trials, size=test_size*2)
    val_indices = set(indices[:len(indices)//2])
    test_indices = set(indices[len(indices)//2:])

    train_gt_labels = set()
    for i in range(num_trials):
        if i in val_indices or i in test_indices:
            continue
        traj_id = traj_idx2id[i]
        gt_label = metadata[traj_id][0]['causorid']
        train_gt_labels.add(gt_label)

    # check all val traj gt labels appear in train
    new_val_indices = val_indices.copy()
    for idx in val_indices:
        traj_id = traj_idx2id[idx]
        gt_label = metadata[traj_id][0]['causorid']
        if gt_label not in train_gt_labels:
            new_val_indices.discard(idx)

    # do same with test indices
    new_test_indices = test_indices.copy()
    for idx in test_indices:
        traj_id = traj_idx2id[idx]
        gt_label = metadata[traj_id][0]['causorid']
        if gt_label not in train_gt_labels:
            new_test_indices.discard(idx)

    # build separate jsons
    val_set = dict()
    for idx in new_val_indices:
        traj_id = traj_idx2id[idx]
        val_set[traj_id] = metadata[traj_id]
    print('val_set size:', len(val_set))

    with open('val_dataset.json', 'w') as f:
        json.dump(val_set, f)

    test_set = dict()
    for idx in new_test_indices:
        traj_id = traj_idx2id[idx]
        test_set[traj_id] = metadata[traj_id]
    print('test_set size:', len(test_set))

    with open('test_dataset.json', 'w') as f:
        json.dump(test_set, f)

    train_set = dict()
    for idx in range(num_trials):
        if idx in new_val_indices or idx in new_test_indices:
            continue
        traj_id = traj_idx2id[idx]
        train_set[traj_id] = metadata[traj_id]
    print('train_set size:', len(train_set))

    with open('train_dataset.json', 'w') as f:
        json.dump(train_set, f)

    # TODO: implement

    return



def send_to(X: Dict[str, torch.tensor], device) -> None:
    """
    sends our input exmaple to device
    Args:
        X:

    Returns:

    """
    X['spectrogram'].to(device)
    X['rgb'].to(device)
    X['gps'].to(device)
    return


def train_loop(model: bl.Tier1BaselineNet,
               data: DataLoader, loss_fn, optimizer, print_freq: int = 100, cuda_device=None) -> None:
    """
    MODIFIES: model
    EFFECTS:  Trains model. Prints current batch loss every print_freq batches
    """
    # closely following https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    size = len(data.dataset)
    for batch, (X, y) in enumerate(data):
        if cuda_device is not None:
            send_to(X, cuda_device)
            y.to(cuda_device)

        pred = model.cuda()(X)
        loss = loss_fn(pred, y.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_freq == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return


def test_loop(model: bl.Tier1BaselineNet, data: DataLoader, loss_fn,
              cuda_device=None) -> (float, float):
    """
    REQUIRES: model not trained on data (i.e. data is validation or test set)
    """
    # closely following https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
    size = len(data.dataset)
    num_batches = len(data)
    correct, test_loss = 0, 0
    with torch.no_grad():
        for X, y in data:
            if cuda_device is not None:
                send_to(X, cuda_device)
                y.to(cuda_device)

            pred = model.cuda()(X)
            #print('test_loop pred.max:', pred.max())
            # fret not, test loss will be averaged across batches
            test_loss += loss_fn(pred, y.cuda()).item()
            correct += (pred.argmax(1) == y.cuda()).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


def train(model: bl.Tier1BaselineNet, data: DataLoader, valdata: DataLoader,
          max_epochs: int = 10000, writer: SummaryWriter = None,
          cuda_device=None, earlystop_threshold: float = 0.05) -> None:
    """
    MODIFIES: model
    """
    model.train()
    # for classification
    loss_fn = nn.CrossEntropyLoss()
    # soundspaces uses Adam defaults, see page 9
    # learning rate 2.5e-4 (but that was for PPO...)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    last_val_loss = np.inf
    for t in range(max_epochs):
        print(f"Epoch {t + 1}\n-------------")
        train_loop(model, data, loss_fn, optimizer, cuda_device=cuda_device)
        current_val_loss, curr_val_acc = test_loop(model, valdata, loss_fn, cuda_device=cuda_device)

        # save for tensorboard
        if writer is not None:
            current_train_loss, curr_train_acc = test_loop(model, data, loss_fn,
                                                           cuda_device=cuda_device)
            writer.add_scalar('training loss', current_train_loss, t)
            writer.add_scalar('training accuracy', curr_train_acc, t)
            writer.add_scalar('validation loss', current_val_loss, t)
            writer.add_scalar('validation accuracy', curr_val_acc, t)

        # early stopping
        if current_val_loss > last_val_loss + earlystop_threshold:
            # TODO: something more sophisticated than stopping now?
            break
        last_val_loss = current_val_loss

    return


def eval(model: bl.Tier1BaselineNet, testdata: DataLoader, cuda_device=None):
    """
    Evaluate the trained model on a test split. Compute accuracy results.
    MODIFIFES: model
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    accuracy, loss = test_loop(model, testdata, loss_fn, cuda_device=cuda_device)
    #num_correct = 0
    #for X, y in testdata:
    #    if cuda_device is not None:
    #        X.to(cuda_device)
    #        y.to(cuda_device)

    #    logits = model(X)
    #    batch_num_correct = (logits.argmax(1) == y).sum()
    #    num_correct += batch_num_correct

    #accuracy = num_correct / len(testdata.dataset)

    # TODO: disambiguation?

    # TODO: top5 accuracy?
    return accuracy, loss


def main():
    rng = np.random.default_rng()
    # TODO: hash, not randint
    run_id = rng.integers(1, 10000, size=1).item()
    writer = SummaryWriter(f'results/run_{run_id}')

    bn = init_model()

    # TODO: test
    max_traj_len = 30
    train_data = Tier1Dataset('data/train_dataset.json', 'data/data_images',
                              'data/convolved_sounds', max_traj_len)
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True,
                                  num_workers=0)
    print(iter(train_dataloader).next())

    # TODO: train model
    #writer.add_graph(bn)
    #train(bn, )

    # TODO: save model and hyperparameters used
    writer.close()
    torch.save(bn, 'baseline.pth')

    return 0


if __name__ == '__main__':
    main()
