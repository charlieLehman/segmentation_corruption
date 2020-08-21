from __future__ import print_function, division
from PIL import Image, ImageOps, ImageFilter
import random
import numpy as np
import torch

class RandomHorizontalFlip(object):
    def __call__(self, img, lbl):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, lbl):
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        lbl = lbl.rotate(rotate_degree, Image.NEAREST)
        return img, lbl

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0, scale=[0.5,2.0]):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.scale = scale

    def __call__(self, img, lbl):
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * self.scale[0]), int(self.base_size * self.scale[1]))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        lbl = lbl.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            lbl = ImageOps.expand(lbl, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        lbl = lbl.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return img, lbl

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.fixscaleresize = FixScaleResize(crop_size)

    def __call__(self, img, lbl):
        w, h = img.size
        img, lbl = self.fixscaleresize(img, lbl)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        lbl = lbl.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return img, lbl

class FixedResize(object):
    def __init__(self, size):
        self.size = size  # size: (h, w)
    def __call__(self, img, lbl):
        w, h = img.size
        img = img.resize((self.size,self.size), Image.BILINEAR)
        lbl = lbl.resize((self.size,self.size), Image.NEAREST)
        return img, lbl


class FixScaleResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl):
        w, h = img.size
        if w > h:
            oh = self.size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        lbl = lbl.resize((ow, oh), Image.NEAREST)
        return img, lbl
