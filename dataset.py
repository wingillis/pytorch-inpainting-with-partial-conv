import cv2
import h5py
import sys
import torch.utils.data as data
import numpy as np
from PIL import Image
from os import listdir
from glob import glob
import os

import torchvision.transforms as transforms


def mouse_transform(im_shape):
    def scale(im):
        return np.uint8(im / 100 * 255)
    def toPIL(im):
        return Image.fromarray(np.tile(im[:, :, None], (1, 1, 3)))
    def _normalize(im):
        return im.mul_(2).add_(-1)

    return transforms.Compose([
        transforms.Lambda(scale),
        transforms.Lambda(toPIL),
        transforms.Resize(im_shape),
        transforms.ToTensor(),
        transforms.Lambda(_normalize),
        transforms.Lambda(lambda a: a.cuda())
        ])


def inverse_mouse_transform(im_shape):
    def scale(im):
        return np.uint16(im / 255 * 100)
    def fromPIL(im):
        return np.array(im)
    def un_normalize(im):
        return im.add_(1).div_(2)
    return transforms.Compose([
        transforms.Lambda(lambda a: a.cpu()),
        transforms.Lambda(un_normalize),
        tansforms.ToPILImage(),
        transforms.Resize(im_shape),
        transforms.Lambda(fromPIL),
        transforms.Lambda(scale)
        ])


def mask_transform(im_shape):
    def scale(im):
        return np.uint8(im * 255)
    def toPIL(im):
        return Image.fromarray(np.tile(im[:, :, None], (1, 1, 3)))
    tsfrm = transforms.Compose([
            transforms.Lambda(scale),
            transforms.Lambda(toPIL),
            transforms.Resize(im_shape, Image.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda a: a.cuda())
        ])
    def run_transform(mouse, mask):
        mouse = mouse < 20
        return tsfrm(((mask > -12) | mouse).astype('uint8'))
    return run_transform
    

class Dataset(data.Dataset):
    def __init__(self, data_path, mask_path, image_shape):
        super(Dataset, self).__init__()
        self.samples = self.find_mouse_images(data_path)
        self.data_path = data_path
        self.mask_files = glob(os.path.join(mask_path, '*.h5'))
        self.image_shape = image_shape
        self.return_name = return_name
        self.imtransform = mouse_transform(self.image_shape)
        self.masktransform = mask_transform(self.image_shape)

    def __getitem__(self, index):
        file_idx = np.where(self.samples > index)[0][0]
        if file_idx > 0:
            index -= self.samples[file_idx - 1]
        with h5py.File(self.mouse_files[file_idx], 'r') as f:
            img = self.imtransform(f['frames'][index])
        mf = np.random.choice(self.mask_files)
        with h5py.File(mf, 'r') as f:
            mask_ind = np.random.randint(f['frames'].shape[0])
            mouse = f['frames'][mask_ind]
            mask = f['frames_mask'][mask_ind]
            mask = self.masktransform(mouse, mask)

        return img * mask, mask, img

    def find_mouse_images(self, dir):
        mouse_files = glob(os.path.join(dir, '*.h5'))
        self.mouse_files = mouse_files
        samples = []
        for m in mouse_files:
            with h5py.File(m, 'r') as f:
                samples += [f['frames'].shape[0]]
        return np.cumsum(samples)

    def __len__(self):
        return self.samples[-1]
