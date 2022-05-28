import h5py
import numpy as np
from supplemental_code import save_obj, detect_landmark
import math
import matplotlib.pyplot as plt
import cv2
import torch


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


if __name__ == '__main__':
    bfm = h5py.File('model2017-1_face12_nomouth.h5', 'r')

    rots = torch.FloatTensor([0, 0, 180])
    t = torch.FloatTensor([0, 0, -500])

    G, color, shape_rep = find_g(bfm)

    img = cv2.imread('beyonce.jpg')[:,:,::-1]
    h, w, _ = img.shape
    gt_landmarks = detect_landmark(img)
    gt_landmarks[:,0] = gt_landmarks[:,0] - w/2
    gt_landmarks[:,1] = gt_landmarks[:,1] - h/2
    
    G_2D = pinhole(G, rots, t, h, w)
    pred_landmarks = get_landmarks(G_2D, plot=False)
    # pred_landmarks = np.loadtxt("Landmarks68_model2017-1_face12_nomouth.anl").astype(int)
    # pred_landmarks = G1[pred_landmarks]

    plt.scatter(pred_landmarks[:,0], pred_landmarks[:,1], color='red', label='prediction')
    plt.scatter(gt_landmarks[:,0], gt_landmarks[:,1], color='blue', label='ground truth')
    plt.legend()
    plt.show()