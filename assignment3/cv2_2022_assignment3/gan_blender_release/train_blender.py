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

import vgg_loss
import discriminators_pix2pix
import res_unet
import gan_loss
from SwappedDataset import SwappedDatasetLoader
import utils
import img_utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb

from blending import *

# wandb.init(project="FSGAN-CV2-project", entity="irenepap")
torch.autograd.set_detect_anomaly(True)
# Configurations
######################################################################
# Fill in your experiment names and the other required components
experiment_name = 'BlenderLaplacian'
data_root = './data_set/data_set/'
train_list = 'train.str'
test_list = 'test.str'
batch_size = 8
nthreads = 4
max_epochs = 20
displayIter = 20
saveIter = 1
img_resolution = 256

lr_gen = 1e-4
lr_dis = 1e-4

momentum = 0.9
weightDecay = 1e-4
step_size = 30
gamma = 0.1

pix_weight = 0.1
rec_weight = 1.0
gan_weight = 0.001

D_PATH = 'Blender/checkpoints/checkpoint_D.pth'
G_PATH = 'Blender/checkpoints/checkpoint_G.pth'
######################################################################
# Independent code. Don't change after this line. All values are automatically
# handled based on the configuration part.

if batch_size < nthreads:
    nthreads = batch_size
check_point_loc = 'Exp_%s/checkpoints/' % experiment_name.replace(' ', '_')
visuals_loc = 'Exp_%s/visuals/' % experiment_name.replace(' ', '_')
os.makedirs(check_point_loc, exist_ok=True)
os.makedirs(visuals_loc, exist_ok=True)
checkpoint_pattern = check_point_loc + 'checkpoint_%s_%d.pth'
logTrain = check_point_loc + 'LogTrain.txt'

torch.backends.cudnn.benchmark = True

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

done = u'\u2713'
print('[I] STATUS: Initiate Network and transfer to device...', end='')
# Define your generators and Discriminators here
discriminator = discriminators_pix2pix.MultiscaleDiscriminator()
generator = res_unet.MultiScaleResUNet(in_nc=7)
# print(done)

print('[I] STATUS: Load Networks...', end='')
# Load your pretrained models here. Pytorch requires you to define the model
# before loading the weights, since the weight files does not contain the model
# definition. Make sure you transfer them to the proper training device. Hint:
    # use the .to(device) function, where device is automatically detected
    # above.
discriminator, _, _ = utils.loadModels(discriminator, path=D_PATH)
generator, _, _ = utils.loadModels(generator, path=G_PATH)
discriminator.to(device)
generator.to(device)
# print(done)

print('[I] STATUS: Initiate optimizer...', end='')
# Define your optimizers and the schedulers and connect the networks from
# before
optimizer_G = torch.optim.SGD(generator.parameters(), momentum=momentum, lr=lr_gen, weight_decay=weightDecay)
optimizer_D = torch.optim.SGD(discriminator.parameters(), momentum=momentum, lr=lr_dis, weight_decay=weightDecay)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)
# print(done)

print('[I] STATUS: Initiate Criterions and transfer to device...', end='')
# Define your criterions here and transfer to the training device. They need to
# be on the same device type.
criterion_pixelwise = nn.L1Loss()
criterion_id = vgg_loss.VGGLoss()
criterion_attr = vgg_loss.VGGLoss()
criterion_gan =gan_loss.GANLoss(use_lsgan=True)

criterion_pixelwise.to(device)
criterion_id.to(device)
criterion_attr.to(device)
criterion_gan.to(device)
# print(done)

print('[I] STATUS: Initiate Dataloaders...', end='')
# Initialize your datasets here
trainDataset = SwappedDatasetLoader(train_list, data_root)
trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
testDataset = SwappedDatasetLoader(test_list, data_root)
testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)
print(iter(testLoader))
# print(done)

print('[I] STATUS: Initiate Logs...', end='')
trainLogger = open(logTrain, 'w')
# print(done)


def transfer_mask(img1, img2, mask):
    return img1 * mask + img2 * (1 - mask)


def visulize_images(source, swap, target, mask, overlaid_image, img_blend, img_blend_alpha, img_blend_lapl):
    f, axarr = plt.subplots(1,4)
    # axarr[0].set_title("Source")
    # axarr[0].imshow(img_utils.tensor2rgb(source[0]))
    # axarr[1].set_title("Swap")
    # axarr[1].imshow(img_utils.tensor2rgb(swap[0]))
    # axarr[2].set_title("Target")
    # axarr[2].imshow(img_utils.tensor2rgb(target[0]))
    # axarr[3].set_title("Mask")
    # axarr[3].imshow(mask[0].permute(1,2,0).squeeze(), cmap='gray')
    axarr[0].set_title("Overlaid")
    axarr[0].imshow(img_utils.tensor2rgb(overlaid_image[0]))
    axarr[1].set_title("Poisson Blending")
    axarr[1].imshow(img_utils.tensor2rgb(img_blend[0]))
    axarr[2].set_title("Alpha Blending")
    axarr[2].imshow(img_utils.tensor2rgb(img_blend_alpha[0]))
    axarr[3].set_title("Laplacian Pyramid Blending")
    # axarr[3].imshow(img_utils.tensor2rgb(img_blend_lapl[0]))
    axarr[3].imshow(img_blend_lapl[0].permute(1,2,0))
    plt.show()


def Train(G, D, epoch_count, iter_count, blend='poisson'):
    G.train(True)
    D.train(True)
    epoch_count += 1
    pbar = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False)

    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1
        images = data

        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        # 2) Feed the data to the networks. 
        # 4) Calculate the losses.
        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.

        # Prepare input
        with torch.no_grad():
            # For each image, push to device
            source = images['source'].to(device) #(fg)
            target = images['target'].to(device) #(bg)
            swap = images['swap'].to(device)     #(sw)
            mask = images['mask'].to(device)     #(mask)   

        # Overlaid image
        overlaid_image = transfer_mask(swap, target, mask)
        # Ground Truth
        img_blend = blend_imgs(overlaid_image, target, mask).to(device)
        # Ground Truth Alpha Blending
        img_blend_alpha = alpha_blending(overlaid_image, target).to(device)
        # Ground Truth Laplacian Blending
        img_blend_lapl = laplacian_blending(target, overlaid_image, mask, num_levels=7).to(device)
        # Concatenate overlaid_image, target and mask to derive the final input
        input = torch.cat((overlaid_image, target, mask), dim=1).to(device)

        # visulize_images(source, swap, target, mask, overlaid_image, img_blend, img_blend_alpha, img_blend_lapl)
        # return None

        # Blend images (pred)
        img_blend_pred = G(input)

        # Fake Detection and Loss
        pred_fake_detached = D(img_blend_pred.detach())
        loss_D_fake = criterion_gan(pred_fake_detached, False)

        # Real Detection and Loss
        pred_real = D(target)
        loss_D_real = criterion_gan(pred_real, True)

        loss_D_total = (loss_D_fake + loss_D_real) * 0.5

        # GAN loss
        pred_fake = D(img_blend_pred)
        loss_G_GAN = criterion_gan(pred_fake, True)

        # Reconstruction
        loss_pixelwise = criterion_pixelwise(img_blend_pred, img_blend)
        loss_id = criterion_id(img_blend_pred, img_blend)
        loss_attr = criterion_attr(img_blend_pred, img_blend)
        loss_rec = pix_weight * loss_pixelwise + 0.5 * loss_id + 0.5 * loss_attr

        loss_G_total = rec_weight * loss_rec + gan_weight * loss_G_GAN

        optimizer_G.zero_grad()
        loss_G_total.backward()
        optimizer_G.step()

        # Update discriminator weights
        optimizer_D.zero_grad()
        loss_D_total.backward()
        optimizer_D.step()

        if iter_count % displayIter == 0:
            # Write to the log file.
            wandb.log((dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), g_gan=loss_G_GAN.item(), d_gan_total=loss_D_total.item(), g_gan_total=loss_G_total.item())))
            trainLogger.write(str(dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), g_gan=loss_G_total.item(), d_gan=loss_D_total.item())))

        # Print out the losses here. Tqdm uses that to automatically print it in front of the progress bar.
        pbar.set_description(str(dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), g_gan=loss_G_total.item(), d_gan=loss_D_total.item())))

    # Save output of the network at the end of each epoch.
    t_source, t_swap, t_target, t_pred, t_blend = Test(G)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)
        wandb.log(dict(images=wandb.Image(grid)))

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    utils.saveModels(D, optimizer_D, iter_count,
                     checkpoint_pattern % ('D', epoch_count))
    tqdm.write('[!] Model Saved!')

    return np.nanmean(loss_pixelwise.cpu().detach().numpy()),\
        np.nanmean(loss_id.cpu().detach().numpy()), np.nanmean(loss_attr.cpu().detach().numpy()),\
        np.nanmean(loss_rec.cpu().detach().numpy()), np.nanmean(loss_G_total.cpu().detach().numpy()),\
        np.nanmean(loss_D_total.cpu().detach().numpy()), iter_count


def Train_without_D(G, epoch_count, iter_count):
    G.train(True)
    epoch_count += 1
    pbar = tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False)

    Epoch_time = time.time()

    for i, data in pbar:
        iter_count += 1
        images = data

        # Implement your training loop here. images will be the datastructure
        # being returned from your dataloader.
        # 1) Load and transfer data to device
        # 2) Feed the data to the networks. 
        # 4) Calculate the losses.
        # 5) Perform backward calculation.
        # 6) Perform the optimizer step.

        # Prepare input
        with torch.no_grad():
            # For each image, push to device
            source = images['source'].to(device) #(fg)
            target = images['target'].to(device) #(bg)
            swap = images['swap'].to(device)     #(sw)
            mask = images['mask'].to(device)     #(mask)   

        # Overlaid image
        overlaid_image = transfer_mask(swap, target, mask)
        # Ground Truth
        img_blend = blend_imgs(overlaid_image, target, mask).to(device)
        # Concatenate overlaid_image, target and mask to derive the final input
        input = torch.cat((overlaid_image, target, mask), dim=1).to(device)

        # Blend images (pred)
        img_blend_pred = G(input)

        # Reconstruction
        loss_pixelwise = criterion_pixelwise(img_blend_pred, img_blend)
        loss_id = criterion_id(img_blend_pred, img_blend)
        loss_attr = criterion_attr(img_blend_pred, img_blend)
        loss_rec = pix_weight * loss_pixelwise + 0.5 * loss_id + 0.5 * loss_attr

        loss_G_total = rec_weight * loss_rec

        optimizer_G.zero_grad()
        loss_G_total.backward()
        optimizer_G.step()

        if iter_count % displayIter == 0:
            # Write to the log file.
            trainLogger.write(str(dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), g_gan=loss_G_total.item())))
            # wandb.log((dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), g_gan_total=loss_G_total.item())))

        # Print out the losses here. Tqdm uses that to automatically print it in front of the progress bar.
        # pbar.set_description(str(dict(g_gan_total=loss_G_total.item())))

    # Save output of the network at the end of each epoch.
    t_source, t_swap, t_target, t_pred, t_blend = Test(G)
    for b in range(t_pred.shape[0]):
        total_grid_load = [t_source[b], t_swap[b], t_target[b],
                           t_pred[b], t_blend[b]]
        grid = img_utils.make_grid(total_grid_load,
                                   cols=len(total_grid_load))
        grid = img_utils.tensor2rgb(grid.detach())
        imageio.imwrite(visuals_loc + '/Epoch_%d_output_%d.png' %
                        (epoch_count, b), grid)
        wandb.log(dict(images=wandb.Image(grid)))

    utils.saveModels(G, optimizer_G, iter_count,
                     checkpoint_pattern % ('G', epoch_count))
    tqdm.write('[!] Model Saved!')

    return np.nanmean(loss_pixelwise.cpu().detach().numpy()),\
        np.nanmean(loss_id.cpu().detach().numpy()), np.nanmean(loss_attr.cpu().detach().numpy()),\
        np.nanmean(loss_rec.cpu().detach().numpy()), np.nanmean(loss_G_total.cpu().detach().numpy()), iter_count


def Test(G):
    with torch.no_grad():
        G.eval()
        t = enumerate(testLoader)
        i, (images) = next(t)

        source = images['source'].to(device) #(fg)
        target = images['target'].to(device) #(bg)
        swap = images['swap'].to(device)     #(sw)
        mask = images['mask'].to(device)     #(mask)   

        # Overlaid image
        overlaid_image = transfer_mask(swap, target, mask)
        # Ground Truth
        blend = blend_imgs(overlaid_image, target, mask).to(device)
        # Feed the network with images from test set
        img_transfer_input = torch.cat((overlaid_image, target, mask), dim=1).to(device)

        # Blend images
        pred = G(img_transfer_input)

        # You want to return 5 components:
        #     1) The source face. (fg)
        #     2) The 3D reconsturction.
        #     3) The target face. (bg)
        #     4) The prediction from the generator. (pred)
        #     5) The GT Blend that the network is targettting. (sw)

        return source, swap, target, pred, blend 


iter_count = 0
# Print out the experiment configurations. You can also save these to a file if
# you want them to be persistent.
print('[*] Beginning Training:')
print('\tMax Epoch: ', max_epochs)
print('\tLogging iter: ', displayIter)
print('\tSaving frequency (per epoch): ', saveIter)
print('\tModels Dumped at: ', check_point_loc)
print('\tVisuals Dumped at: ', visuals_loc)
print('\tExperiment Name: ', experiment_name)

for i in range(max_epochs):
    # Call the Train function here
    # Step through the schedulers if using them.
    # You can also print out the losses of the network here to keep track of epoch wise loss.
    
    # Train GAN
    loss_pixelwise, loss_id, loss_attr, loss_rec, loss_G_total, loss_D_total, iter_count= Train(generator, discriminator, i, iter_count)
    # wandb.log((dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), d_gan_total=loss_D_total.item(), g_gan_total=loss_G_total.item())))
    # print(str(dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), g_gan=loss_G_GAN.item(), d_gan=loss_D_total.item())))
    
    # Train without discriminator
    # loss_pixelwise, loss_id, loss_attr, loss_rec, loss_G_total, iter_count= Train_without_D(generator, i, iter_count)
    # wandb.log((dict(pixelwise=loss_pixelwise.item(), id=loss_id.item(), attr=loss_attr.item(), rec=loss_rec.item(), g_gan_total=loss_G_total.item())))

trainLogger.close()
