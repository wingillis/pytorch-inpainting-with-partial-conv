import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import h5py
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from dataset import Dataset as Places2
import dataset
from util.io import load_ckpt
from util.io import save_ckpt
from util.image import unnormalize


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--apply', type=str)
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

model = PConvUNet().to(device)

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

if args.apply:
    start_iter = load_ckpt(
        args.apply, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

model.eval()

imtransform = dataset.mouse_transform((256, 256))
masktransform = dataset.mask_transform((256, 256))
inversetransform = dataset.inverse_transform((80, 80))
# TODO: compose difference flows for evaluation and training 
grayscale = transforms.Grayscale(1)
toPIL = transforms.ToPILImage()

with h5py.File(args.root, 'r+') as f:
    frames, mask = f['frames'][:1000], f['frames_mask'][:1000]
    output = []
    for i in range(1000):
        _im = imtransform(frames[i])
        _ma = masktransform(frames[i], mask[i])
        res, _ = model(_im.view(1, *_im.size()), _ma.view(1, *_ma.size()))
        # res = _ma * _im + (1 - _ma) * res
        # output += [np.array(grayscale(toPIL(unnormalize(res[0]))))]
        tmp = inversetransform(res)
        output += [tmp]
    f.create_dataset('inpaint', data=np.array(output))
