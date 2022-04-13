import numpy as np
import open3d as o3d
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import cKDTree

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.


######                                                           ######
##      notice: This is just some example, feel free to adapt        ##
######                                                           ######


# == Load data ==
def open3d_example():
    pcd = o3d.io.read_point_cloud("Data/data/0000000000.pcd")
    # ## convert into ndarray

    pcd_arr = np.asarray(pcd.points)

    # ***  you need to clean the point cloud using a threshold ***
    pcd_arr_cleaned = pcd_arr

    # visualization from ndarray
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)
    o3d.visualization.draw_geometries([vis_pcd])


def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy'))
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy'))
    return source, target


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

############################
#     ICP                  #
############################

def icp(A1, A2, max_iterations=20, epsilon=0.01, kd_tree=False):

    ###### 0. (adding noise)

    ###### 1. initialize R= I , t= 0
    R = rotation = np.identity(3)
    t = translation = np.zeros((3,1))
    source = A1

    past_rms = np.inf

    for iter in range(max_iterations):
        # go to 2. unless RMS is unchanged(<= epsilon)

        # 2. using different sampling methods
        
        # 3. transform point cloud with R and t
        A1 = R @ A1 + t
        # plot_progress(source, A2, A1, iter)

        # 4. Find the closest point for each point in A1 based on A2 using brute-force approach
        if kd_tree:
            matches = calculate_closest_points_kd_tree(A1, A2)
        else:
            matches = calculate_closest_points(A1, A2)

        # 5. Calculate RMS
        rms = calculate_RMS(A1, matches)

        # check if RMS is unchanged or within threshold
        print('Iter', iter, 'RMS', np.abs(rms-past_rms))
        if abs(past_rms-rms) < epsilon:
            break
        
        past_rms = rms

        # 6. Refine R and t using SVD
        R, t = get_new_R_t(A1, matches)

        # Update rotation and translation
        rotation = R @ rotation
        translation = R @ translation + t

    return rotation, translation


def plot_progress(source, target, trans, iter, dir='./figures/waves', save_figure=True):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(trans[0], trans[1], trans[2], label='Transformation')
    ax.scatter3D(target[0], target[1], target[2], label='Target')
    ax.scatter3D(source[0], source[1], source[2], label='Source')
    ax.legend()
    if save_figure:
        plt.savefig(dir + f'/prog_{iter}.png')
    else:
        plt.show()


############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.


############################
#  Additional Improvements #
############################

if __name__ == "__main__":
    # open3d_example()
    # source, target = open_wave_data()
    source, target = open_bunny_data()
    # plot_progress(source, target, source, './figures/waves')
    R, t = icp(source, target, kd_tree=True, epsilon=0.00001, max_iterations=50)
    trans = (R @ source) + t
    plot_progress(source, target, trans, iter=0, save_figure=False)
