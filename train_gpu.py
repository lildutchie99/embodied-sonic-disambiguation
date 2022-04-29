"""
train_parallel.py

train our model on the cluster

EECS 692 project
WN 22 April 2022
SPA, Nikhil, Ayush, Ben
"""

import time
import argparse
import logging

from trainer import *
from model.main import init_model
import torch.cuda
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablate",
        type=str,
        choices=["none", "vision", "sound", "gps"],
        # required=True,
        default='none',
        help="which sensor to ablate",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        default='../../ayshrv/',
        help="path to data folder containing train_dataset.json, val_dataset.json, and test_dataset.json",
    )
    parser.add_argument(
        "--imgdir",
        type=str,
        required=True,
        default='../../ayshrv/shortest_path_example',
        help='path to folder containing all images (directories with trajid, each containign imgs)',
    )
    parser.add_argument(
        "--sounddir",
        type=str,
        required=True,
        default='../../ayshrv/convolved_sounds',
        help='path to folder containing all .wav files'
    )
    parser.add_argument(
        "--checkpoint",
        type=bool,
        default=False,
        help="whether to save checkpoints",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results",
        help="path to where tensorboard results are saved"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=100,
        help="batch size for training"
    )
    parser.add_argument(
        "--maxepochs",
        type=int,
        default=100000,
        help="maximum number of passes through training data to train"
    )
    parser.add_argument(
        "--earlystop-threshold",
        type=float,
        default=0.05,
        help="slack param for early stopping, validation Cross Entropy Loss"
    )
    parser.add_argument(
        "--eval",
        type=bool,
        default=True,
        help="whether to run forward pass on test set after training"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cpu', 'gpu'],
        default='gpu',
        help="whether to use cpu or gpu"
    )
    parser.add_argument(
        "--maxtrajlen",
        type=int,
        default=30,
        help="maximum trajectory length (timesteps) across all 3 datasets"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    # todo: more elegant
    ablate_sound, ablate_vision, ablate_gps = False, False, False
    if args.ablate == 'sound':
        ablate_sound = True
    elif args.ablate == 'vision':
        ablate_vision = True
    elif args.ablate == 'gps':
        ablate_gps = True

    # tensorboard automatically assigns a run id based on current time and machine...
    writer = SummaryWriter()
    allargs = vars(args)
    writer.add_text('training_params', str(allargs))

    NUM_WORKERS = 0  # TODO: should we parallelize this?

    # load data
    ANNOTATIONS_FILENAME = 'dataset.json'
    img_dir = args.imgdir
    sound_dir = args.sounddir
    objidmappingpath = args.metadata + '/objid_to_index_17DRP5sb8fy.json'
    train_data = Tier1Dataset(args.metadata + '/train_' + ANNOTATIONS_FILENAME, img_dir,
                              sound_dir, args.maxtrajlen, objidmappingpath)
    val_data = Tier1Dataset(args.metadata + '/val_' + ANNOTATIONS_FILENAME, img_dir, sound_dir,
                            args.maxtrajlen, objidmappingpath)
    test_data = Tier1Dataset(args.metadata + '/test_' + ANNOTATIONS_FILENAME, img_dir, sound_dir,
                             args.maxtrajlen, objidmappingpath)
    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True,
                                  num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_data, batch_size=args.batchsize, shuffle=True,
                                num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True,
                                 num_workers=NUM_WORKERS)

    bn = init_model(ablate_sound=ablate_sound, ablate_vision=ablate_vision, ablate_gps=ablate_gps)
    #writer.add_graph(bn, iter(train_dataloader).next())

    if not torch.cuda.is_available():
        print('cuda not available, using cpu')

    #device = torch.device('cuda' if args.device == 'gpu' else 'cpu')
    # FIXME: ensure device is cuda
    device = 'cuda'
    bn = bn.cuda()

    train(bn, train_dataloader, val_dataloader, max_epochs=args.maxepochs,
          writer=writer, cuda_device=device, earlystop_threshold=args.earlystop_threshold)

    if args.eval:
        test_accuracy, test_loss = eval(bn, test_dataloader, cuda_device=device)
        writer.add_text('test_accuracy', str(test_accuracy))
        writer.add_text('test_crossentropy_loss', str(test_loss))

    # TODO: match model save name to tensorboard writer run Id
    model_save_name = '-'.join(time.ctime().replace(':', '-').split())
    torch.save(bn, f'saved_models/model_{model_save_name}.pth')

    writer.close()

    return 0


if __name__ == '__main__':
    main()
