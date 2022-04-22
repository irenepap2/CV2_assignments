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


def calculate_closest_points_z_buffer(A1, A2, H=100, W=100, m=5):
    '''
    Computes closest points for every source pointcloud A1 using the z-buffer
    '''

    # Even m results in unclear window.
    if m % 2 == 0:
        raise ValueError("Value m={} invalid. m must be odd.".format(m))

    # calculate floor of m / 2
    m_flr = m // 2

    # Take the union of A1, A2.
    union = np.vstack((A1.T, A2.T))
    union = union.T


    # Determine the minimum enclosing box on the x,y-plane.
    box = {"x_min": np.min(union[0]), "x_max": np.max(union[0]),
           "y_min": np.min(union[1]), "y_max": np.max(union[1])}

    width = box["x_max"] - box["x_min"]
    w_step = width / W

    height = box["y_max"] - box["y_min"]
    h_step = height / H


    # Divide minimum enclosing box into H×W rectangular cells and find centroid for each cell
    cell_centroids = np.zeros((H*W, 2))
    for i in range(H):
        for j in range(W):
            ind = i * H + j
            cell_centroids[ind][0] = box['x_min'] + w_step/2 + j*w_step
            cell_centroids[ind][1] = box['y_min'] + h_step/2 + i*h_step


    # Calculate distance from each point to each centroid
    kdTree = cKDTree(cell_centroids)
    _, idx_source = kdTree.query(A1[:2].T)
    _, idx_target = kdTree.query(A2[:2].T)


    # Buffers with (HxW, 2) to store index and z-value repectively
    source_buffer = np.zeros((H*W, 2))
    for i, idx in enumerate(idx_source):
        if np.abs(A1[2, i]) > np.abs(source_buffer[idx][1]):
            source_buffer[idx][0] = i
            source_buffer[idx][1] = A1[2, i]

    target_buffer = np.zeros((H*W, 2))
    for i, idx in enumerate(idx_target):
        if np.abs(A2[2, i]) > np.abs(target_buffer[idx][1]):
            target_buffer[idx][0] = i
            target_buffer[idx][1] = A2[2, i]


    # Find matches between source and target
    matches = {'source': [], 'target': []}
    for i in range(H*W):
        closest = (i, np.inf)

        # Save closest target cell in m*m area
        for y in range(i - m_flr*W, i + m_flr*(W + 1), W):
            for x in range(y - m_flr, y + m_flr + 1):

                # Prevent x from indexing out-of-bounds.
                if x < 0 or x > H*W - 1:
                    continue
                elif np.abs(target_buffer[x][1]) < np.abs(closest[1]):
                    closest = (x, target_buffer[x][1])

        matches['source'].append(i)
        matches['target'].append(closest[0])


    # Slice A1, A2 to obtain matching subsets.
    new_A1 = A1[:, matches['source']]
    matches = A2[:, matches['target']]

    return new_A1, matches


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
    U, _, Vt = np.linalg.svd(A)

    I = np.eye(Vt.shape[1])
    I[-1, -1] = 1/np.linalg.det(Vt.T @ U.T)

    R = Vt.T @ I @ U.T

    t = y_centroid - R @ x_centroid
    t = np.expand_dims(t, axis=1)

    return R, t


def icp(A1, A2, sampling='none', max_iters=50, epsilon=1e-4, ratio=0.1, mode='kd_tree', N=4, alpha=0.3, noise=False, noise_max=0.1, print_rms=True):
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
        - mode:        ['kd_tree', 'z_buffer']. Determine whether kd_tree algorithm
                       or z_buffer is used for point matching.
        - N:           Determines step size of multi-res.
        - alpha:       Set the alpha value for the triangle mesh.
        - noise        Set whether noise is added.
        - noise_max    Set the amount of noise.
        - print_rms    Print the RMS each iter if true.

    output:
        - rotation:    Final rotation.
        - translation: Final translation.
        - all_rms:     RMS value at each iteration.
    '''

    ###### 0. (adding noise)
    if noise:
        A1 = gauss_noise(A1, noise_max)
        A2 = gauss_noise(A1, noise_max)

    ###### 1. initialize R= I , t= 0
    R = rotation = np.identity(3)
    t = translation = np.zeros((3,1))

    past_rms = np.inf
    all_rms = []

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
            new_A1 = R @ new_A1 + t

        # No sampling (brute force)
        else:
            new_A1 = A1
            new_A2 = A2

        # Use KDTree to find distances
        if mode == 'kd_tree':
            matches = calculate_closest_points_kd_tree(new_A1, new_A2)
        # Use Z-buffer to match points
        elif mode == 'z_buffer':
            new_A1, matches = calculate_closest_points_z_buffer(new_A1, new_A2)
        else:
            matches = calculate_closest_points(new_A1, new_A2)

        # 4. Calculate RMS
        rms = calculate_RMS(new_A1, matches)
        all_rms.append(rms)

        if print_rms:
            print('Iter', iter, 'RMS', rms)

        # 5. Check if RMS is unchanged or within threshold
        if abs(past_rms-rms) < epsilon:
            break

        past_rms_x = past_rms
        past_rms = rms

        # 6. Refine R and t using SVD
        R, t = get_new_R_t(new_A1, matches)

        # Move in the sampling amount dependent on RMS for multi-res.
        if sampling == 'multi_res':
            # Increase points sampled
            if cur_step != len(steps) - 1 and np.abs(rms-past_rms_x) < steps[cur_step]:
                cur_step += 1
                new_A1 = subsample_graph(A1, points=points_sampled[cur_step])
                if points_sampled[cur_step] <= A2.shape[1]:
                    new_A2 = subsample_graph(A2, points=points_sampled[cur_step])
                else:
                    new_A2 = subsample_graph(A2, points=A2.shape[1])


        # 7. Update rotation and translation
        rotation = R @ rotation
        translation = R @ translation + t

    return rotation, translation, all_rms


############################
#  Additional Improvements #
############################


if __name__ == "__main__":

    # source, target, save_dir = open_wave_data()
    source, target, save_dir = open_bunny_data()
    samplings = ['uniform', 'random', 'multi_res', 'info_reg', 'none']
    time1 = time.time()
    R, t, _ = icp(source, target, sampling='none', epsilon=1e-8, max_iters=50, ratio=0.1, mode='kd_tree')
    print("Time:", time.time() - time1)
    trans = (R @ source) + t
    plot_progress(source, target, trans, dir=save_dir, save_figure=False)
