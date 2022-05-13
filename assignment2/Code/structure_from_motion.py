from fundamental_matrix import *
from pointview_matrix import *
import cv2 as cv
import numpy as np
from scipy import spatial
import open3d as o3d
from utils import *


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


def compute_structures(pvm, frame_step=3, one_block=False):
    structures, idxs = [], []

    for i in range(0, len(pvm)-frame_step*2, 2):

        # Select xy pairs for frame_step frames.
        block = pvm[i:i+frame_step*2, :]
        # Create dense blocks
        p = np.all(block != -1.0, axis=0)
        block = block[:, p]
        # keep track of the block indices to find their intersection later on
        idx = np.where(p)

        # Do not consider blocks that are < 3 frames or < 3 points per frame.
        if block.shape[0] != frame_step*2 or block.shape[1] < frame_step:
            continue

        idxs.append(idx[0])

        # Normalise block.
        block = block - np.expand_dims(np.mean(block, axis=1), axis=1)

        S = find_structure_motion(block)
        structures.append(S)
        if one_block:
            return S

    return structures, idxs


def visualize(S):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(S.T)
    o3d.visualization.draw_geometries([pcd])


def factorize_and_stich(structs, index_list):

    merged_cloud = []
    N = len(structs)

    for i in range(N-1):
        # find the index intersection of the two structures
        indx_intersection = np.intersect1d(index_list[i], index_list[i + 1])

        # keep only the indices that are in the intersection
        indx1 = [int(np.where(index_list[i] == x)[0]) for x in indx_intersection]
        indx2 = [int(np.where(index_list[i + 1] == x)[0]) for x in indx_intersection]

        s1 = structs[i][:, indx1]       
        s2 = structs[i + 1][:, indx2]

        if i == 0:
            _, _, _, R, s, norm1, norm2 = procrustes(s1.T, s2.T)
            trans = (s2.T - np.mean(s2.T, axis=0)) @ R.T * s
            merged_cloud = np.vstack((trans/norm1, structs[i + 1].T/norm2))
        else:
            _, _, _, R, s, norm1, norm2 = procrustes(s1.T, s2.T)
            merged_cloud = (merged_cloud - np.mean(merged_cloud, axis=0)) @ R.T * s
            merged_cloud = np.vstack((merged_cloud/norm1, structs[i + 1].T/norm2))

    return merged_cloud


if __name__ == '__main__':

    pvm = np.loadtxt('PVM_ours_last.txt')
    # pvm = np.loadtxt('PointViewMatrix.txt')

    # Change frame_step to set how many consecutive frames
    # structs, index_list = compute_structures(pvm, frame_step=4)
    # merged_pcd = factorize_and_stich(structs, index_list)
    # scale z axis for better visualization
    # merged_pcd[:, 2] = merged_pcd[:, 2] * 5
    # visualize(merged_pcd.T)

    # visualize one block
    S = compute_structures(pvm, frame_step=3, one_block=True).T
    visualize(S.T)

    #visualize their PointViewMatrix.txt
    pvm = np.loadtxt('PointViewMatrix.txt')
    pvm = pvm - np.expand_dims(np.mean(pvm, axis=1), axis=1)
    S = find_structure_motion(pvm)
    visualize(S)
