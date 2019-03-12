import argparse
import numpy as np
import os
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import h5py
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
import dataset
from util.io import load_ckpt
from util.io import save_ckpt
from util.image import unnormalize
from dls_net.util import generate_indices, add_h5_dataset


parser = argparse.ArgumentParser()
# testing options
parser.add_argument('--root', type=str, default='/n/groups/datta/win')
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--image-size', type=int, default=256)
parser.add_argument('--apply', type=str)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
if args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = PConvUNet().to(device)

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()))

if args.apply:
    start_iter = load_ckpt(
        args.apply, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4

model.eval()

imtransform = dataset.mouse_transform((256, 256), training=False, cuda=args.cuda)
masktransform = dataset.mask_transform((256, 256), training=False, cuda=args.cuda)
inversetransform = dataset.inverse_mouse_transform((80, 80))


with h5py.File(args.root, 'r') as f:
    frames = f['frames'][()]
    output = np.zeros(frames.shape, dtype=np.uint8)
    mask = f['frames_mask'][()]
    total = frames.shape[0] // 8
    for start, end in tqdm(generate_indices(8, frames.shape[0]), desc='applying model', total=total):
        _im = torch.stack([imtransform(x) for x in frames[start:end]])
        _ma = torch.stack([masktransform(x, m) for x, m in zip(frames[start:end], mask[start:end])])
        res, _ = model(_im, _ma)
        tmp = [inversetransform(x) for x in res]
        output[start:end, ...] = np.array(tmp)
add_h5_dataset(args.root, 'inpaint', data=output)