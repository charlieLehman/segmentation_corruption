from torchvision.utils import save_image
import argparse
from tqdm import tqdm
import numpy as np
import torch
import os

from imagenet_c import corrupt, corruption_tuple
from functools import partial
from itertools import product as iterprod
from PIL import Image
from torchvision import transforms
import joint_transforms

from torchvision.datasets import VOCSegmentation

corr_dict = {}
[corr_dict.update({p.__name__.split('_')[0]:n}) for n,p in enumerate(corruption_tuple[:15])]

MEAN_STD = {"mean":(0.485, 0.456, 0.406), "std":(0.229, 0.224, 0.225)}

base_size = 224
crop_size = 224
class ImLblCorruptTransform(object):
    def __init__(self, severity, corruption_number):
        corrupt_partial = partial(corrupt, severity=severity, corruption_number=corruption_number)
        self.joint_transform = joint_transforms.FixedResize(224)
        self.transform = lambda sz: transforms.Compose(
            [
            np.array,
            corrupt_partial,
            Image.fromarray,
            transforms.Resize(sz),
            transforms.ToTensor(), 
            ]
        )
        if severity == 0:
            self.transform = lambda sz: transforms.Compose(
                [
                transforms.ToTensor(), 
                ]
            )
    def __call__(self, img, lbl):
        img, lbl = self.joint_transform(img,lbl)
        W,H = img.size
        sz = (H,W)
        img = self.transform(sz)(img)
        return img, lbl


def main():
    parser = argparse.ArgumentParser(description='Dump voc c')
    parser.add_argument('--cn', type=int, default=4, metavar='N',
                        help='Corruption Number')
    parser.add_argument('--sv', type=int, default=1, metavar='N',
                        help='Severity')

    args=parser.parse_args()
    sv = args.sv
    corruption_name = corruption_tuple[args.cn].__name__
    if args.cn==-1 or sv==-1:
        if not os.path.isdir('VOC-C/lbl'):
            os.mkdir('VOC-C/lbl')
    if not os.path.isdir('VOC-C/{}'.format(corruption_name)):
        os.mkdir('VOC-C/{}'.format(corruption_name))
    if not os.path.isdir('VOC-C/{}/{}'.format(corruption_name,sv)):
        os.mkdir('VOC-C/{}/{}'.format(corruption_name,sv))
    corr_val = VOCSegmentation(root='/data/datasets/',
                            transforms=ImLblCorruptTransform(sv,args.cn),
                            image_set='val')
    iterator = enumerate(tqdm(corr_val))
    for n, (im,lbl) in iterator:
        if args.cn==-1 or sv==-1:
            lbl.save('VOC-C/lbl/{:04d}.png'.format(n))
        else:
            save_image(im, 'VOC-C/{}/{}/{:04d}.png'.format(corruption_name,sv,n))

if __name__ == '__main__':
    main()
