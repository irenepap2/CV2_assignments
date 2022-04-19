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
    '''
    Solve for R, t using Singular Value Decomposition
    '''
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


def icp(A1, A2, sampling=[], max_iters=50, epsilon=1e-4, ratio=0.1, kd_tree=False, N=4, alpha=0.3, noise=False, noise_max=0.1):
    '''
    Perform ICP algorithm to perform Rototranslation.

    input:
        - A1:          The source image.
        - A2:          The target image.
        - sampling:    The sampling type of ['uniform', 'random', 'multi_res',
                                            'info_reg', 'none'].
        - max_iters:   Max iterations the ICP can run.
        - epsilon:     Error threshold.
        - ratio:       Total % of points to use for sampling.
        - kd_tree:     Determine whether kd_tree algorithm used for
                       point matching.
        - N:           Determines step size of multi-res.
        - alpha:       Set the alpha value for the triangle mesh.
        - noise        Set whether noise is added.
        - noise_max    Set the amount of noise.

    output:
        - rotation:    Final rotation.
        - translation: Final translation.
    '''

    ###### 0. (adding noise)

    ###### 1. initialize R= I , t= 0
    R = rotation = np.identity(3)
    t = translation = np.zeros((3,1))

    past_rms = np.inf
  
    if sampling == 'uniform':
        new_A1 = subsample_graph(A1, points=int(A1.shape[1] * ratio))
        new_A2 = subsample_graph(A2, points=int(A2.shape[1] * ratio))
        
    if sampling == 'info_reg':
        new_A1, new_A2 = obtain_informative_regions(A1, A2, alpha, int(A1.shape[1] * ratio))

    if sampling == 'multi_res':
        points_sampled, steps = set_multi_res(A1, epsilon, N)
        cur_step = 0
        new_A1 = subsample_graph(A1, points=points_sampled[cur_step])
        new_A2 = subsample_graph(A2, points=points_sampled[cur_step])

    for iter in range(max_iters):
        
        # 2. transform point cloud with R and t
        A1 = R @ A1 + t

        # 3. Find the closest point for each point in A1 based on A2 using different approaches

        # Uniform sampling and sampling based on informative regions.
        if sampling == 'uniform' or sampling == 'info_reg':
            new_A1 = R @ new_A1 + t

        # Random sampling
        elif sampling == 'random':
            new_A1 = subsample_graph(A1, points=int(A1.shape[1] * ratio))
            new_A2 = subsample_graph(A2, points=int(A2.shape[1] * ratio))
        
        # Multi resolution
        elif sampling == 'multi_res':
            new_A1 = subsample_graph(A1, points=points_sampled[cur_step])
            if points_sampled[cur_step] <= A2.shape[1]:
                new_A2 = subsample_graph(A2, points=points_sampled[cur_step])
            else:
                new_A2 = subsample_graph(A2, points=A2.shape[1])

        # No sampling (brute force)
        else:
            new_A1 = A1
            new_A2 = A2
        
        if kd_tree:
            matches = calculate_closest_points_kd_tree(new_A1, new_A2)
        else:
            matches = calculate_closest_points(new_A1, new_A2)

        # 4. Calculate RMS
        rms = calculate_RMS(new_A1, matches)
        
        # Move in the sampling amount dependent on RMS for multi-res.
        if sampling == 'multi_res':
            # Increase points sampled
            while cur_step != len(steps) - 1 and np.abs(rms-past_rms) < steps[cur_step]:
                cur_step += 1

        print('Iter', iter, 'RMS', rms)
        # 5. Check if RMS is unchanged or within threshold
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

    # source = subsample_graph(source, points=int(source.shape[1] * 0.85))
    # target = subsample_graph(target, points=int(source.shape[1] * 0.85))
    
    samplings = ['uniform', 'random', 'multi_res', 'info_reg', 'none']
    time1 = time.time()
    R, t = icp(source, target, sampling=samplings[4], epsilon=1e-8, max_iters=50, ratio=0.1, kd_tree=False)
    print("Time:", time.time() - time1)
    trans = (R @ source) + t
    plot_progress(source, target, trans, file_path='./figures/wave.png', save_figure=False)
