""" The main function for the rPPG deep learning pipeline.

TODO: Adds detailed description for models and datasets supported.
An end-to-end training pipleine for neural network methods.

  Typical usage example:

  python main.py --model_name physnet --data_dir "G:\\UBFC_data"
"""

import argparse
import glob
import os
import time
from torch.utils.data import DataLoader
from dataset.data_loader.data_loader import data_loader
from dataset.data_loader.UBFC_loader import UBFC_loader
from dataset.data_loader.COHFACE_loader import COHFACE_loader
from dataset.data_loader.PURE_loader import PURE_loader
from tensorboardX import SummaryWriter
from neural_methods.trainer.trainer import trainer
from neural_methods.trainer.physnet_trainer import physnet_trainer


def get_UBFC_data(args):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/UBFC_dataloader.py """
    data_dirs = glob.glob(args.data_dir + os.sep + "subject*")
    return {
        "train": data_dirs[:-2],
        "valid": data_dirs[-2:-1],
        "test": data_dirs[-1:]
    }


def get_COHFACE_data(args):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/COHFACE_dataloader.py """
    data_dirs = glob.glob(args.data_dir + os.sep + "*")
    return {
        "train": data_dirs[:-2],
        "valid": data_dirs[-2:-1],
        "test": data_dirs[-1:]
    }


def get_PURE_data(args):
    """Returns directories for train sets, validation sets and test sets.
    For the dataset structure, see dataset/dataloader/PURE_dataloader.py """
    data_dirs = glob.glob(args.data_dir + os.sep + "*-*")
    return {
        "train": data_dirs[:-2],
        "valid": data_dirs[-2:-1],
        "test": data_dirs[-1:]
    }

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--model_name', default="physnet", type=str, help="The name of the model.")
    parser.add_argument('--dataset', required=True, choices=["COHFACE","PURE","UBFC"], type=str, help="The Dataset. Supporting UBFC/PURE/COHFACE.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument(
        '--device',
        default=0,
        type=int,
        help="An integer to specify which gpu to use, -1 for cpu.")
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--data_dir', required=True,
                        type=str, help='The path of the data directory.')
    current_time = time.localtime()
    parser.add_argument('--name',
                        default="{0}-{1}".format(str(current_time.tm_hour),
                                                 str(current_time.tm_min)),
                        type=str)
    parser.add_argument('--round_num_max', default=20, type=int)
    return parser


def main(args, writer, data_loader):
    """Trains the model."""
    trainer_name = eval('{0}_trainer'.format(args.model_name))
    trainner = trainer_name(args, writer)
    trainner.train(data_loader)


if __name__ == "__main__":
    # parses arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.add_trainer_args(parser)
    parser = data_loader.add_data_loader_args(parser)
    args = parser.parse_args()
    writer = SummaryWriter('runs/exp/' + args.name)

    # loads data
    data_files = eval("get_{0}_data".format(args.dataset))(args)
    loader = eval("{0}_loader".format(args.dataset))
    train_data = loader(
        data_dirs=data_files["train"],
        cached_dir=args.cached_dir)
    valid_data = loader(
        data_dirs=data_files["valid"],
        cached_dir=args.cached_dir)
    test_data = loader(
        data_dirs=data_files["test"],
        cached_dir=args.cached_dir)
    dataloader = {
        "train": DataLoader(
            dataset=train_data,
            num_workers=2,
            batch_size=args.batch_size,
            shuffle=True),
        "valid": DataLoader(
            dataset=valid_data,
            num_workers=2,
            batch_size=args.batch_size,
            shuffle=True),
        "test": DataLoader(dataset=test_data, num_workers=2,
                           batch_size=args.batch_size, shuffle=True)
    }
    main(args, writer, dataloader)
