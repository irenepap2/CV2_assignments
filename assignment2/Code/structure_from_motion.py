from fundamental_matrix import *
from pointview_matrix import *
import cv2 as cv
import numpy as np
from scipy import spatial


def find_structure_motion(block):
    U, W, VT = np.linalg.svd(block)

    # Take three columns, values, rows respectively.
    U = U[:, :3]
    W = np.diag(W[:3])
    VT = VT[:3, :]

    # Calculate Motion and Structure matrices.
    M = U @ np.sqrt(W)
    S = np.sqrt(W) @ VT

    # Check if rank of D = M @ S is equal to 3, which it should always be.
    assert np.linalg.matrix_rank(M @ S) == 3

    return S


def compute_structures(pvm, frame_step=3):
    structures, idxs = [], []
    for i in range(0, len(pvm), frame_step*2):

        # Select xy pairs for frame_step frames.
        block = pvm[i:i+frame_step*2, :]

        p = np.all(block != -1.0, axis=0)
        block = block[:, p]
        idx = np.where(p)

        # Remove all colums that contain at least one -1 entry.
        # block = block[:, np.all(block != -1.0, axis=0)]

        # Do not consider blocks that are < 3 frames or < 3 points per frame.
        if block.shape[0] != frame_step*2 or block.shape[1] < frame_step:
            continue

        idxs.append(idx)

        # Normalise block.
        block = block - np.expand_dims(np.mean(block, axis=1), axis=1)

        S = find_structure_motion(block)
        structures.append(S)

    return structures, idxs


if __name__ == '__main__':

    pvm = np.loadtxt('PVM_ours.txt')
    # pvm = create_pv_matrix()
    visualize_pvm(pvm.copy())
    # mtx2 = shared_points(mtx)
    # print(mtx2.shape)
    # print(mtx2)
    structs, _ = compute_structures(pvm)
    print(len(structs))
    lens = [s.shape for s in structs]
    print(lens)