import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import imageio

import res_unet
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils

from blending import *

def transfer_mask(img1, img2, mask):
    return img1 * mask + img2 * (1 - mask)

def Test(G, testLoader, device):
    pbar = tqdm(enumerate(testLoader), total=len(testLoader), leave=False)

    with torch.no_grad():
        G.eval()
        for i, images in pbar:

            source = images['source'].to(device) #(fg)
            target = images['target'].to(device) #(bg)
            swap = images['swap'].to(device)     #(sw)
            mask = images['mask'].to(device)     #(mask)   

            # Overlaid image
            overlaid_image = transfer_mask(swap, target, mask)
            # Feed the network with images from test set
            img_transfer_input = torch.cat((overlaid_image, target, mask), dim=1).to(device)

            # Blend images
            pred = G(img_transfer_input)

            for b in range(source.shape[0]):
                img = img_utils.tensor2rgb(source[b])
                imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d_%d.png' % (20, i, b), img)


if __name__ == '__main__':

    cudaDevice = ''

    if len(cudaDevice) < 1:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('[*] GPU Device selected as default execution device.')
        else:
            device = torch.device('cpu')
            print('[X] WARN: No GPU Devices found on the system! Using the CPU. '
                'Execution maybe slow!')
    else:
        device = torch.device('cuda:%s' % cudaDevice)
        print('[*] GPU Device %s selected as default execution device.' %
            cudaDevice)

    G_PATH = 'Exp_BlenderOriginal/checkpoints/checkpoint_G_20.pth'

    test_list = 'test.str'
    data_root = './data_set/data_set/'
    batch_size = 8
    testDataset = SwappedDatasetLoader(test_list, data_root)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)

    generator = res_unet.MultiScaleResUNet(in_nc=7)
    generator, _, _ = utils.loadModels(generator, path=G_PATH)

    generator.to(device)

    experiment_name = 'BlenderOriginal'
    visuals_loc = 'Exp_%s/source_imgs/' % experiment_name.replace(' ', '_')
    os.makedirs(visuals_loc, exist_ok=True)

    Test(generator, testLoader, device)
   