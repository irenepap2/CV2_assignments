import h5py
import numpy as np
from supplemental_code import save_obj, detect_landmark, render
import math
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from energy_optimization import *
from os.path import exists
import imutils


def find_g(bfm, alpha=None, delta=None, save_G=False, name="morphable2.obj"):
    shape_mean = np.asarray(bfm['shape/model/mean'], dtype=np.float32)
    shape_mean = np.reshape(shape_mean, (-1, 3))

    shape_base = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
    shape_base = np.reshape(shape_base, (shape_base.shape[0] // 3, 3, -1))[:,:,:30]

    shape_var = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)[:30]
    shape_var = np.sqrt(shape_var)

    exp_mean = np.asarray(bfm['expression/model/mean'], dtype=np.float32)
    exp_mean = np.reshape(exp_mean, (-1, 3))

    exp_base = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
    exp_base = np.reshape(exp_base, (exp_base.shape[0] // 3, 3, -1))[:,:,:20]

    exp_var = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)[:20]
    exp_var = np.sqrt(exp_var)

    if alpha is None:
        alpha = torch.FloatTensor(30).uniform_(-1, 1)
    if delta is None:
        delta = torch.FloatTensor(20).uniform_(-1, 1)

    shape_mean = torch.from_numpy(shape_mean)
    shape_base = torch.from_numpy(shape_base)
    shape_var = torch.from_numpy(shape_var)
    exp_mean =  torch.from_numpy(exp_mean)
    exp_base = torch.from_numpy(exp_base)
    exp_var =  torch.from_numpy(exp_var)

    G = shape_mean + shape_base @ (shape_var * alpha) + exp_mean + exp_base @ (exp_var * delta)

    color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    color = np.reshape(color, (color.shape[0] // 3, 3))

    shape_rep = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
    if save_G:
        save_obj(name, G, color, shape_rep)

    return G, color, shape_rep


def get_landmarks(G_3D, plot=False):
    with open("Landmarks68_model2017-1_face12_nomouth.anl", 'r') as f:
        idxs = f.read().split('\n')
        idxs = [int(x) for x in idxs]
        landmarks = G_3D[idxs, :2]

        if plot:
            plt.scatter(landmarks[:, 0], landmarks[:, 1])
            plt.show()

        return landmarks


def pinhole(G, Rots, t, h, w, fov=0.5):

    Rx, Ry, Rz = Rots
    Rx, Ry, Rz = math.radians(Rx), math.radians(Ry), math.radians(Rz)
    Rotx = np.array([[1, 0, 0],
                    [0, np.cos(Rx), -np.sin(Rx)],
                    [0, np.sin(Rx), np.cos(Rx)]])
    Roty = np.array([[np.cos(Ry), 0, np.sin(Ry)],
                    [0, 1, 0],
                    [-np.sin(Ry), 0, np.cos(Ry)]])
    Rotz = np.array([[np.cos(Rz), -np.sin(Rz), 0],
                    [np.sin(Rz), np.cos(Rz), 0],
                    [0, 0, 1]])

    R = (torch.from_numpy(Rotx) @ torch.from_numpy(Roty) @ torch.from_numpy(Rotz)).float()

    bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')
    color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    color = np.reshape(color, (color.shape[0] // 3, 3))

    G = G @ R.T + t

    # shape_rep = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
    # save_obj("morphable_rot.obj", G_R, color, shape_rep)

    #convert to homogeneous coordinates
    G_ = torch.cat((G, torch.ones(G.shape[0],1)), dim=1)

    # Find ranges
    aspect_ratio = w/h
    vn = 600
    vf = 1000
    vt = np.tan(fov/2) * vn
    vb = - vt
    vr = vt * aspect_ratio
    vl = -vt * aspect_ratio

    V = np.array([[(vr-vl)/2, 0, 0, (vr+vl)/2],
                  [0, (vt-vb)/2, 0, (vt+vb)/2],
                  [0, 0, 1/2, 1/2],
                  [0, 0, 0, 1]])

    P = np.array([[2*vn/(vr-vl), 0, (vr+vl)/(vr-vl), 0],
                 [0, 2*vn/(vt-vb), (vt+vb)/(vt-vb), 0],
                 [0, 0, -(vf+vn)/(vf-vn), -2*vf*vn/(vf-vn)],
                 [0, 0, -1, 0]])

    Pi = torch.from_numpy(V @ P).float()

    G_3D = (G_ @ Pi.T)

    # Dividing by homogenous coordinate.
    G_3D = (G_3D / G_3D[:, 3][:, None])

    return G_3D[:, :3]


def texturing(img, model, bfm, h, w, interp=True):
    gt_landmarks = detect_landmark(img)
    gt_landmarks[:, 0] = gt_landmarks[:, 0]
    gt_landmarks[:, 1] = gt_landmarks[:, 1]

    G_3D, G_2D, pred_landmarks, shape_rep = model(bfm, h, w)
    G_2D = G_2D[:, :2]

    G_3D[:, 0] = G_3D[:, 0] + w/2
    G_3D[:, 1] = G_3D[:, 1] - h/2
    G_3D[:, 1] = G_3D[:, 1] * -1  # flip horizontally

    G_2D[:, 0] = G_2D[:, 0] + h/2
    G_2D[:, 1] = G_2D[:, 1] - w/2

    img_rot = imutils.rotate(img, 90)
    colors = []
    for x, y in G_2D:
        if interp:
            c = interpolate(x.item(), y.item(), img_rot)
        else:
            x_flr, y_flr = int(np.floor(x.item())), int(np.floor(y.item()))
            c = img_rot[x_flr][y_flr]

        colors.append(c)

    colors = np.array(colors)
    textured = render(G_3D.detach().numpy(), colors / 255, shape_rep, H=h, W=w)

    return colors, textured


def interpolate(x, y, img):
    # Prevent division by 0 errors
    if x.is_integer():
        x += 0.0001
    if y.is_integer():
        y += 0.0001

    x1, x2 = int(np.floor(x)), int(np.ceil(x))
    y1, y2 = int(np.floor(y)), int(np.ceil(y))

    c_11 = img[x1][y1]
    c_12 = img[x1][y2]
    c_21 = img[x2][y2]
    c_22 = img[x2][y2]

    # Linear interpolation in x-direction
    c_xy1 = (x2 - x) / (x2 - x1) * c_11 + (x - x1) / (x2 - x1) * c_21
    c_xy2 = (x2 - x) / (x2 - x1) * c_12 + (x - x1) / (x2 - x1) * c_22

    # Linear interpolation in y-direction
    c_xy = (y2 - y) / (y2 - y1) * c_xy1 + (y - y1) / (y2 - y1) * c_xy2

    return c_xy


def multiple_frames(imgs, bfm):
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        gt_landmarks = detect_landmark(img)
        gt_landmarks[:,0] = gt_landmarks[:,0] - w/2
        gt_landmarks[:,1] = gt_landmarks[:,1] - h/2

        model = EnergyOptim()
        model, _ = train_model(model, bfm, torch.from_numpy(gt_landmarks), h, w, iters=10000)

        G_3D, _, _, shape_rep = model(bfm, h, w)
        color, _ = texturing(img, model, bfm, h, w)

        save_obj(f"./results_images/4.2.5/mesh_sep_{i}.obj", G_3D.detach().numpy(), color, shape_rep)


def faceswap(img1, img2, bfm):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Generate img1 texture
    gt_landmarks1 = detect_landmark(img1)
    gt_landmarks1[:, 0] = gt_landmarks1[:, 0] - w1/2
    gt_landmarks1[:, 1] = gt_landmarks1[:, 1] - h1/2

    model = EnergyOptim()
    model, _ = train_model(model, bfm, torch.from_numpy(gt_landmarks1), h1, w1, iters=10000)
    color, _ = texturing(img1, model, bfm, h1, w1)


    # Generate img2 geometry
    gt_landmarks2 = detect_landmark(img2)
    gt_landmarks2[:, 0] = gt_landmarks2[:, 0] - w2/2
    gt_landmarks2[:, 1] = gt_landmarks2[:, 1] - h2/2

    model = EnergyOptim()
    model, _ = train_model(model, bfm, torch.from_numpy(gt_landmarks2), h2, w2, iters=10000)
    G_3D, _, _, shape_rep2 = model(bfm, h2, w2)

    G_3D[:, 0] = G_3D[:, 0] * 2.1 + w2/2
    G_3D[:, 1] = G_3D[:, 1] * 2.1 - h2/2.4
    G_3D[:, 1] = G_3D[:, 1] * -1  # flip horizontally

    # Perform face swap.
    face_map = render(G_3D.detach().numpy(), color / 255, shape_rep2, h2, w2)
    swap = np.where(face_map, face_map, img2 / 255)

    return swap



if __name__ == '__main__':
    bfm = h5py.File('model2017-1_face12_nomouth.h5', 'r')

    # rots = torch.FloatTensor([0, 0, 180])
    # t = torch.FloatTensor([0, 0, -500])

    # G, color, shape_rep = find_g(bfm)

    img = cv2.imread('beyonce.jpg')[:,:,::-1]
    # h, w = img.shape[:2]
    # gt_landmarks = detect_landmark(img)
    # gt_landmarks[:,0] = gt_landmarks[:,0] - w/2
    # gt_landmarks[:,1] = gt_landmarks[:,1] - h/2

    # G_2D = pinhole(G, rots, t, h, w)
    # pred_landmarks = get_landmarks(G_2D, plot=False)
    # pred_landmarks = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl").astype(int)
    # pred_landmarks = G1[pred_landmarks]

    # if exists('./energy_optim_model.pt'):
    #     model = EnergyOptim()
    #     model.load_state_dict(torch.load('./energy_optim_model.pt'))
    # else:
    #     model = EnergyOptim()
    #     model, losses = train_model(model, bfm, torch.from_numpy(gt_landmarks), h, w)
    #     torch.save(model.state_dict(), './energy_optim_model.pt')


    # # Gt and untrained on ref img
    # plt.scatter(gt_landmarks[:,0] + 0.5*w, gt_landmarks[:,1] + 0.5*h, color='blue', label='ground truth')
    # plt.scatter(pred_landmarks[:,0] + .5*w, pred_landmarks[:,1] + 0.5*h, color='red', label='predicted')
    # plt.imshow(img)
    # plt.legend()
    # plt.show()

    # _, _, pred_landmarks = model(bfm, h, w)
    # pred_landmarks = pred_landmarks.detach().numpy()

    # _, textured = texturing(img, model, bfm, h, w)
    # plt.imshow(textured)
    # plt.show()

    # # Gt and trained
    # plt.scatter(pred_landmarks[:,0], pred_landmarks[:,1], color='red', label='prediction')
    # plt.scatter(gt_landmarks[:,0], gt_landmarks[:,1], color='blue', label='ground truth')
    # plt.legend()
    # plt.show()

    # # Gt and trained on ref img
    # plt.scatter(gt_landmarks[:,0] + 0.5*w, gt_landmarks[:,1] + 0.5*h, color='blue', label='ground truth')
    # plt.scatter(pred_landmarks[:,0] + .5*w, pred_landmarks[:,1] + 0.5*h, color='red', label='predicted')
    # plt.imshow(img)
    # plt.legend()
    # plt.show()

    img2 = cv2.imread('beyonce2.png')[:,:,::-1]
    img3 = cv2.imread('beyonce3.png')[:,:,::-1]

    multiple_frames([img, img2, img3], bfm)

    swift = cv2.imread('swift.jpg')[:,:,::-1]
    swap = faceswap(img, swift, bfm)
    plt.imshow(swap)
    plt.show()
