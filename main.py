import argparse
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from experiment import Experiment
import os
import faulthandler
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
faulthandler.enable()


parser = argparse.ArgumentParser(description='Acquire some parameters for fusion restore')
parser.add_argument('--lamda_gp', type=float, default=10, help='lambda for wgan gp')
parser.add_argument('--glr', type=float, default=2e-3,
                    help='the initial learning rate')
parser.add_argument('--dlr', type=float, default=2e-3,
                    help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')

parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('.'),
                    help='the output directory')

parser.add_argument('--data_dir', type=Path, default='CIA',
                    help='the training data directory')
parser.add_argument('--image_size', type=int, nargs='+',default=[2040, 1720],
                    help='the image size (height, width)')

parser.add_argument('--patch_stride', type=int, nargs='+', default=200,
                    help='the patch stride for training')
parser.add_argument('--patch_size', type=int, nargs='+', default=256,
                    help='the patch size for prediction')
opt = parser.parse_args()

torch.manual_seed(2020)
if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    torch.cuda.manual_seed_all(2020)
    cudnn.benchmark = True
    cudnn.deterministic = True

opt.patch_size = opt.image_size if opt.patch_size is None else opt.patch_size

if __name__ == '__main__':
    experiment = Experiment(opt)
    train_dir = opt.data_dir / 'train'
    val_dir = opt.data_dir / 'val'
    test_dir = val_dir
    if opt.epochs > 0:
        if opt.epochs > 0:
            experiment.train(train_dir, val_dir, opt.patch_stride, opt.batch_size, num_workers=opt.num_workers, epochs=opt.epochs)

    experiment.test(test_dir, opt.patch_size, num_workers=opt.num_workers)