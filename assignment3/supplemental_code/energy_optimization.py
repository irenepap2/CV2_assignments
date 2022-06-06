import torch
import math
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from face_swapping import *
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm


def find_g(bfm, alpha=None, delta=None, save_G=False, name="morphable2.obj"):
    '''
    Obtain point cloud, triangles and colors of BFM.
    '''

    # PCA for shape
    shape_mean = np.asarray(bfm['shape/model/mean'], dtype=np.float32)
    shape_mean = np.reshape(shape_mean, (-1, 3))

    shape_base = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
    shape_base = np.reshape(shape_base, (shape_base.shape[0] // 3, 3, -1))[:,:,:30]

    shape_var = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)[:30]
    shape_var = np.sqrt(shape_var)

    # PCA for expression
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
    exp_mean = torch.from_numpy(exp_mean)
    exp_base = torch.from_numpy(exp_base)
    exp_var = torch.from_numpy(exp_var)

    G = shape_mean + shape_base @ (shape_var * alpha) + exp_mean + exp_base @ (exp_var * delta)

    color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    color = np.reshape(color, (color.shape[0] // 3, 3))

    shape_rep = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
    if save_G:
        save_obj(name, G, color, shape_rep)

    return G, color, shape_rep


def pinhole(G, Rots, t, h, w, fov=0.5, save_model=False, name="rotation.obj", shape_rep=None):
    '''
    Calcualte and apply the pinhole camera of a point cloud with rotation and
    translation.
    '''

    # Create rotation matrix
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

    # Roto translate the model
    G = G @ R.T + t

    # Save model if wanted
    if save_model:
        save_obj(name, G, color, shape_rep)

    # Convert to homogeneous coordinates
    G_ = torch.cat((G, torch.ones(G.shape[0],1)), dim=1)

    # Find ranges
    aspect_ratio = w/h
    vn = 600
    vf = 1000
    vt = np.tan(fov/2) * vn
    vb = - vt
    vr = vt * aspect_ratio
    vl = -vt * aspect_ratio

    # Create viewport matrix
    V = np.array([[(vr-vl)/2, 0, 0, (vr+vl)/2],
                  [0, (vt-vb)/2, 0, (vt+vb)/2],
                  [0, 0, 1/2, 1/2],
                  [0, 0, 0, 1]])
    # Create perspective projection matrix
    P = np.array([[2*vn/(vr-vl), 0, (vr+vl)/(vr-vl), 0],
                 [0, 2*vn/(vt-vb), (vt+vb)/(vt-vb), 0],
                 [0, 0, -(vf+vn)/(vf-vn), -2*vf*vn/(vf-vn)],
                 [0, 0, -1, 0]])

    # Create the pinhole camera and apply.
    Pi = torch.from_numpy(V @ P).float()

    G_3D = (G_ @ Pi.T)

    # Dividing by homogenous coordinate.
    G_3D = (G_3D / G_3D[:, 3][:, None])

    return G_3D[:, :3]


def get_landmarks(G_3D, plot=False):
    '''
    Obtain and possibly plot the landmark points of 3D point cloud.
    '''
    with open("Landmarks68_model2017-1_face12_nomouth.anl", 'r') as f:
        idxs = f.read().split('\n')
        idxs = [int(x) for x in idxs]
        landmarks = G_3D[idxs, :2]

        if plot:
            plt.scatter(landmarks[:, 0], landmarks[:, 1])
            plt.show()

        return landmarks


class EnergyOptim(nn.Module):
    '''
    Neural net for the parameter optimization.
    '''
    def __init__(self):

        super(EnergyOptim, self).__init__()

        self.alpha = nn.Parameter(torch.FloatTensor(30).uniform_(-1, 1), requires_grad=True)
        self.delta = nn.Parameter(torch.FloatTensor(20).uniform_(-1, 1), requires_grad=True)
        self.Rots = nn.Parameter(torch.FloatTensor([0, 0, 180]), requires_grad=True)
        self.t = nn.Parameter(torch.FloatTensor([0, 0, -500]), requires_grad=True)

    def forward(self, bfm, h, w):
        # Find 3D face representation.
        face_3D, _, shape_rep = find_g(bfm, alpha=self.alpha, delta=self.delta)
        # Project onto 2D using pinhole and predict landmark points.
        face_2D = pinhole(face_3D, self.Rots, self.t, h, w)
        pred = get_landmarks(face_2D)

        return (face_3D, face_2D, pred, shape_rep)


def loss_c(landmarks, pred, alpha, delta, l_alpha=0, l_delta=0):
    '''
    Calculate Energy minimization loss.
    '''
    Llan = torch.sum(torch.square(pred - landmarks)) / 68
    Lreg = l_alpha * torch.sum(torch.square(alpha)) + l_delta * torch.sum(torch.square(delta))
    return (Llan + Lreg)


def train_model(model, bfm, gt, h, w, iters=10000):
    '''
    Return a trained model.
    '''
    # Set the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    losses = []
    # Training loop.
    for i in range(iters):
        # Predict and calculate loss.
        _, _, pred, _ = model(bfm, h, w)
        loss = loss_c(gt, pred, model.alpha, model.delta)
        losses.append(loss.item())

        # Model step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print every 1000 iterations.
        if i % 1000 == 0:
            print(f"Finished iter {i}")
            print('loss:', loss)

    return(model, losses)


if __name__ == '__main__':
    '''
    Run the training of the parameters and plot first and final prediction
    vs the ground truth.
    '''
    # Load bfm
    bfm = h5py.File('model2017-1_face12_nomouth.h5', 'r')

    img = cv2.imread('beyonce.jpg')[:, :, ::-1]  # Convert from BGR to RGB
    h, w, _ = img.shape

    # Get gt landmarks
    gt_landmarks = detect_landmark(img)
    gt_landmarks[:,0] = gt_landmarks[:,0] - w/2
    gt_landmarks[:,1] = gt_landmarks[:,1] - h/2

    # Create and train model.
    model = EnergyOptim()
    model, losses = train_model(model, bfm, torch.from_numpy(gt_landmarks), h, w)
    print(model.alpha, "\n")
    print(model.delta, "\n")
    print(model.Rots, "\n")
    print(model.t, "\n")

    # Plot final prediction.
    _, _, pred_landmarks = model(bfm, h, w).detach().numpy()
    print(pred_landmarks.shape)
    plt.scatter(pred_landmarks[:,0], pred_landmarks[:,1], color='red', label='prediction')
    plt.scatter(gt_landmarks[:,0], gt_landmarks[:,1], color='blue', label='ground truth')
    plt.legend()
    plt.show()
