from tkinter import N
import numpy as np
import img_utils
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def blend_imgs_bgr(source_img, target_img, mask):
    # Implement poisson blending here. You can us the built-in seamlessclone
    # function in opencv which is an implementation of Poisson Blending.
    a = np.where(mask != 0)
    if len(a[0]) == 0 or len(a[1]) == 0:
        return target_img
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return target_img

    center = (np.min(a[1]) + np.max(a[1])) // 2, (np.min(a[0]) + np.max(a[0])) // 2
    output = cv2.seamlessClone(source_img, target_img, mask, center, cv2.NORMAL_CLONE)

    return output


def blend_imgs(source_tensor, target_tensor, mask_tensor):
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        mask = np.round(mask * 255).astype('uint8')
        # Poisson blending
        out_bgr = blend_imgs_bgr(source_img, target_img, mask)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)


def alpha_blending(source_tensor, target_tensor, alpha=0.5):
    # Implements alpha blending
    out_tensors = []
    beta = 1.0 - alpha
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2bgr(source_tensor[b])
        target_img = img_utils.tensor2bgr(target_tensor[b])
        out_bgr = cv2.addWeighted(source_img, alpha, target_img, beta, 0)
        out_tensors.append(img_utils.bgr2tensor(out_bgr))

    return torch.cat(out_tensors, dim=0)


# Find the Gaussian pyramid of the two images and the mask
def gaussian_pyramid(img, num_levels):
    """
    https://theailearner.com/tag/image-blending-with-pyramid-and-mask/
    """

    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    """
    https://theailearner.com/tag/image-blending-with-pyramid-and-mask/
    """

    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1
    
    laplacian_pyr = [laplacian_top]
    for i in range(num_levels,0,-1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def blend(laplacian_A, laplacian_B, mask_pyr):
    """
    https://theailearner.com/tag/image-blending-with-pyramid-and-mask/
    """

    LS = []
    for la,lb,mask in zip(laplacian_A, laplacian_B, mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    """
    https://theailearner.com/tag/image-blending-with-pyramid-and-mask/
    """

    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


def laplacian_blending(source_tensor, target_tensor, mask_tensor, num_levels = 2):
    """
    https://theailearner.com/tag/image-blending-with-pyramid-and-mask/
    """

    # Implements laplacian blending
    out_tensors = []
    for b in range(source_tensor.shape[0]):
        source_img = img_utils.tensor2rgb_without_rounding(source_tensor[b])
        target_img = img_utils.tensor2rgb_without_rounding(target_tensor[b])
        # source_img = img_utils.tensor2rgb(source_tensor[b])
        # target_img = img_utils.tensor2rgb(target_tensor[b])

        # Turn to float32
        source_img = np.float32(source_img)
        target_img = np.float32(target_img)

        mask = mask_tensor[b].permute(1, 2, 0).cpu().numpy()
        mask = np.concatenate([mask, mask, mask], axis=-1)
        mask = np.float32(mask)

        # For image-1, calculate Gaussian and Laplacian
        gaussian_pyr_1 = gaussian_pyramid(source_img, num_levels)
        laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)

        # For image-2, calculate Gaussian and Laplacian
        gaussian_pyr_2 = gaussian_pyramid(target_img, num_levels)
        laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)

        # Calculate the Gaussian pyramid for the mask image and reverse it.
        mask_pyr_final = gaussian_pyramid(mask, num_levels)
        mask_pyr_final.reverse()

        # Blend the images
        add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
        # Reconstruct the images
        out_rgb = reconstruct(add_laplace)

        # out_tensors.append(img_utils.rgb2tensor(out_rgb[num_levels]))
        out_tensors.append(torch.from_numpy(out_rgb[num_levels]).permute(2, 0, 1).unsqueeze(0))

    return torch.cat(out_tensors, dim=0)