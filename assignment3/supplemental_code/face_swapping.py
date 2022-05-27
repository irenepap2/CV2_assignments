import h5py
import numpy as np
from supplemental_code import *


def find_g(save_G=False, name="morphable2.obj"):
    bfm = h5py.File("model2017-1_face12_nomouth.h5", 'r')
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

    alpha1 = np.random.uniform(-1, 1, size=(30))
    alpha2 = np.random.uniform(-1, 1, size=(20))

    G = shape_mean + shape_base @ (shape_var * alpha1) + exp_mean + exp_base @ (exp_var * alpha2)

    color = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    color = np.reshape(color, (color.shape[0] // 3, 3))

    shape_rep = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
    if save_G:
        save_obj(name, G, color, shape_rep)
    return G


def pinhole(G, Rot, type='Ox'):
    "NOTHING HERE MAKES SENSE"
    if type == 'Ox':
        R = np.array([[1, 0, 0],
                      [0, np.cos(Rot), -np.sin(Rot)],
                      [0, np.sin(Rot), np.cos(Rot)]])
    elif type == 'Oy':
        R = np.array([[np.cos(Rot), 0, np.sin(Rot)],
                      [0, 1, 0],
                      [-np.sin(Rot), 0, np.cos(Rot)]])
    elif type == 'Oz':
        R = np.array([[np.cos(Rot), -np.sin(Rot), 0],
                      [np.sin(Rot), np.cos(Rot), 0],
                      [0, 0, 1]])
    G_R = G @ R.t
    G_ = np.c_[G_R, np.ones(G.shape[0])]

    S = np.array([[2/2, 0, 0, 0],
                [0, 1/2, 0, 0],
                [0, 0, 2/2, 0],
                [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, 2/2],
                [0, 1, 0, 1/2],
                [0, 0, 1, 2/2],
                [0, 0, 0, 1]])

    V = T @ S

    return


if __name__ == '__main__':
    G = find_g(save_G=True)
    # G_ = np.c_[G, np.ones(G.shape[0])]
    pinhole(G, 10, type='Oy')
    # print(G_.shape)
