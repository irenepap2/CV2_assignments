import numpy as np
from scipy.spatial.distance import cdist
import time
from scipy.spatial import cKDTree
from utils import *

############################
#     ICP                  #
############################

def calculate_closest_points(A1, A2):
    '''  
    Computes euclidean distance from every source pointcloud A1 
    to every target pointcloud A2 and returns the corresponding A2 matches 
    with the minimum distance from the A1 cloudpoints
    '''

    euclidean_distances = cdist(A1.T, A2.T)
    shortest_indx = np.argmin(euclidean_distances, axis=1)
    matches = A2[:, shortest_indx]

    return matches


def calculate_closest_points_kd_tree(A1, A2):
    '''  
    Computes closest points for every source pointcloud A1 using the kd-tree
    '''

    tree = cKDTree(A2.T)
    _, shortest_indx = tree.query(A1.T) 
    matches = A2[:, shortest_indx]

    return matches


def calculate_RMS(source, target):
    rms = np.sqrt(np.mean(((source - target)**2)))
    return rms


def get_new_R_t(source, target):

    x_centroid = np.mean(source, axis=1)
    y_centroid = np.mean(target, axis=1)

    #compute the centered vectors
    x = source.T - x_centroid
    y = target.T - y_centroid
    
    #compute the covariance matix
    A = x.T @ y

    #compute the SVD and get R,t
    U, S, Vt = np.linalg.svd(A)

    I = np.eye(Vt.shape[1])
    I[-1, -1] = 1/np.linalg.det(Vt.T @ U.T)

    R = Vt.T @ I @ U.T
    
    t = y_centroid - R @ x_centroid
    t = np.expand_dims(t, axis=1)

    return R, t


def icp(A1, A2, sampling=[], max_iters=50, epsilon=1e-4, total_p=1000, kd_tree=False):
    '''
    Perform ICP algorithm to perform Rototranslation.
    
    input:
        - A1:          The source image.
        - A2:          The target image.
        - sampling:    The sampling type of ['uniform', 'random', 'multi_res',
                                            'info_reg', 'none'].
        - max_iters:   Max iterations the ICP can run.
        - epsilon:     Error threshold.
        - dist:        Total points to use for sampling.
        - kd_tree:     Determine whether kd_tree algorithm used for
                       point matching.
        - N:           Something with multiRes TODO.
    
    output:
        - rotation:    Final rotation.
        - translation: Final translation.
    '''

    ###### 0. (adding noise)

    ###### 1. initialize R= I , t= 0
    R = rotation = np.identity(3)
    t = translation = np.zeros((3,1))

    past_rms = 1
    
    if sampling == 'uniform':
        new_A1 = subsample_graph(A1, points=total_p)
        new_A2 = subsample_graph(A2, points=total_p)

    for iter in range(max_iters):
        
        # 2. transform point cloud with R and t
        A1 = R @ A1 + t

        # 3. Find the closest point for each point in A1 based on A2 using different approaches

        # Uniform sampling
        if sampling == 'uniform':
            new_A1 = R @ new_A1 + t

        # Random sampling
        elif sampling == 'random':
            new_A1 = subsample_graph(A1, points=total_p)
            new_A2 = subsample_graph(A2, points=total_p)

        # No sampling (brute force)
        else:
            new_A1 = A1
            new_A2 = A2
        
        if kd_tree:
            matches = calculate_closest_points_kd_tree(new_A1, new_A2)
        else:
            matches = calculate_closest_points(new_A1, new_A2)

        # 4. Calculate RMS
        if sampling and sampling != 'none':
            rms = calculate_RMS(new_A1, matches)
        else:
            rms = calculate_RMS(A1, matches)

        # 5. Check if RMS is unchanged or within threshold
        print('Iter', iter, 'RMS', rms)
        if abs(past_rms-rms) < epsilon:
            break
        
        past_rms = rms

        # 6. Refine R and t using SVD
        R, t = get_new_R_t(new_A1, matches)

        # 7. Update rotation and translation
        rotation = R @ rotation
        translation = R @ translation + t

    return rotation, translation


############################
#  Additional Improvements #
############################

if __name__ == "__main__":

    # source, target = open_wave_data()
    source, target = open_bunny_data()
    
    samplings = ['uniform', 'random', 'multi_res', 'info_reg', 'none']
    time1 = time.time()
    R, t = icp(source, target, sampling=samplings[1], epsilon=1e-10, max_iters=50, total_p=source.shape[1] // 100, kd_tree=True)
    print("Time:", time.time() - time1)
    trans = (R @ source) + t
    plot_progress(source, target, trans, dir='./figures/bunny', save_figure=False)
