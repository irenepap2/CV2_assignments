import numpy as np
import open3d as o3d
import os
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import cKDTree
from random import sample

# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.
FRAMES_PATH = DATA_DIR + '/data/'

# == Load data ==
def get_cleaned_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    
    # convert into ndarray
    pcd_arr = np.asarray(pcd.points)

    # clean the point cloud using a threshold
    dist = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[dist < 2]

    return pcd_arr_cleaned


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

    past_rms = 1

    for iter in range(max_iterations):
        # go to 2. unless RMS is unchanged(<= epsilon)

        # 2. using different sampling methods
        
        # 3. transform point cloud with R and t
        A1 = R @ A1 + t

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
    
    # visualization from ndarray
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source.T)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target.T)
    trans_pcd = o3d.geometry.PointCloud()
    trans_pcd.points = o3d.utility.Vector3dVector(trans.T)
    source_pcd.paint_uniform_color([1, 0, 0])
    target_pcd.paint_uniform_color([0, 1, 0])
    trans_pcd.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source_pcd, target_pcd, trans_pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source_pcd)
    vis.add_geometry(target_pcd)
    vis.add_geometry(trans_pcd)
    vis.poll_events()
    vis.update_renderer()

    if save_figure:
        vis.capture_screen_image(dir + f'/prog_{iter}.png')


############################
#   Merge Scene            #
############################

def get_frame_pointclouds(step=1, frame_count=100):
    '''  
    Returns frame pointclouds depending on step size
    step = N = {1,2,4,10}
    '''

    frames = []
 
    counts = ["{0:02}".format(i) for i in range(0, frame_count, step)]
    filename_paths = [FRAMES_PATH + f'00000000{i}.pcd' for i in counts]

    for file_path in filename_paths:
        frames.append(get_cleaned_pointcloud(file_path).T)

    return frames


def estimate_transformations(frame_pcds):
    
    #  Estimate the camera poses using two consecutive frames of given data.
    rotations = []
    translations = []
    for i in range(len(frame_pcds)-1):
        print('Frame ', i+1)
        R, t = icp(frame_pcds[i], frame_pcds[i+1], kd_tree=True, epsilon=1e-4, max_iterations=50)
        rotations.append(R)
        translations.append(t)

    # #Estimating rotation and translation from last frame to first
    # print('Frame ', len(frame_pcds)-1)
    # R, t = icp(frame_pcds[len(frame_pcds)-1], frame_pcds[0], kd_tree=True, epsilon=1e-4, max_iterations=50)
    # rotations.append(R)
    # translations.append(t)
    
    return rotations, translations


############################
#  Additional Improvements #
############################

def subsample_graph(A1, points=1000):
    '''
    Create reduced point cloud for subsampling
    '''
    gen_points = sample(range(0, len(A1[0] - 1)), points)
    new_A1 = np.array([[], [], []])

    # Create downsampled A1 from generated points
    for point in gen_points:
        new_point = np.array([[A1[0][point]], [A1[1][point]], [A1[2][point]]])
        new_A1 = np.hstack((new_A1, new_point))

    return np.array(new_A1)


def merge_scene(merge_in_between=False, frame_count=100, step=1, visualize=True):
    '''
    Estimates the camera poses of every N frames and merges the results
    (either in between every ipc calculation, or all together at the end)
    '''

    frame_pcds = get_frame_pointclouds(step=step, frame_count=frame_count) # step = N = {1, 2, 4, 10}

    #  Iteratively merge and estimate the camera poses for the consecutive frames.
    if merge_in_between:
        source_frame = frame_pcds[0]
        for i, frame in enumerate(frame_pcds[1:]):
            print("Frame:", i+1)
            R, t = icp(source_frame, frame, kd_tree=True, epsilon=1e-2, max_iterations=10)
            new_source = R @ source_frame + t
            source_frame = np.hstack((new_source, frame))
        
        trans = subsample_graph(source_frame, points=20000)
        trans_pcd = o3d.geometry.PointCloud()
        trans_pcd.points = o3d.utility.Vector3dVector(trans.T)
        trans_pcds = [trans_pcd]

    # Estimate the camera poses for the consecutive frames and merge at the end
    else:
        rotations, translations = estimate_transformations(frame_pcds) 
        
        trans_pcds = []
        for i, frame in enumerate(frame_pcds):
            r_list = rotations[i:]
            t_list = translations[i:]
            for r, t in zip(r_list, t_list):
                frame = r @ frame + t

            trans = subsample_graph(frame)
            trans_pcd = o3d.geometry.PointCloud()
            trans_pcd.points = o3d.utility.Vector3dVector(trans.T)
            trans_pcds.append(trans_pcd)

    if visualize:
        o3d.visualization.draw_geometries(trans_pcds)

if __name__ == "__main__":

    merge_scene()

    # source, target = open_bunny_data()
    # # source, target = open_wave_data()
    # R, t = icp(source, target, kd_tree=True, epsilon=1e-8, max_iterations=50)
    # trans = (R @ source) + t
    # plot_progress(source, target, trans, iter='last', dir='./figures/bunny', save_figure=True)
