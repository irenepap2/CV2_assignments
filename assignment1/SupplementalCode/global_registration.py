import numpy as np
import open3d as o3d
from utils import *
from icp import *

############################
#   Merge Scene            #
############################

def estimate_transformations(frame_pcds, epsilon, max_iters, mode):
    
    #  Estimate the camera poses using two consecutive frames of given data.
    rotations = []
    translations = []
    for i in range(len(frame_pcds)-1):
        print('Frame ', i+1)
        R, t, _ = icp(frame_pcds[i], frame_pcds[i+1], mode=mode, epsilon=epsilon, max_iters=max_iters)
        rotations.append(R)
        translations.append(t)
    
    return rotations, translations


def visualize(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    o3d.visualization.draw_geometries([pcd])


def merge_scene_in_between(frame_pcds, epsilon=1e-7, max_iters=100):
    '''
    Estimates the camera poses and merges every N frames
    '''

    #  Iteratively merge and estimate the camera poses for the consecutive frames.
    merged_pcd = frame_pcds[0]
    for i, frame in enumerate(frame_pcds[1:]):
        print("Frame:", i+1)
        R, t, _ = icp(merged_pcd, frame, sampling='multi_res', mode='kd_tree', epsilon=epsilon, max_iters=max_iters)
        new_source = R @ merged_pcd + t
        merged_pcd = np.hstack((new_source, frame))
    
    return merged_pcd


def merge_scene_end(frame_pcds, epsilon=1e-7, max_iters=100):
    '''
    Estimates the camera poses of every N frames and merges the results at the end
    '''

    # Estimate the camera poses for the consecutive frames and merge at the end
    for i in range(len(frame_pcds)-1):
        print("Frame:", i+1)
        source = frame_pcds[i]
        target = frame_pcds[i+1]
        R, t, _ = icp(source, target, mode='kd_tree', epsilon=epsilon, max_iters=max_iters)
        trans = R @ source + t
        if i == 0:
            merged_pcd = trans
        else:
            merged_pcd = R @ merged_pcd + t
            merged_pcd = np.hstack((merged_pcd, trans))
    
    #last rototranslation from last frame to first frame
    print("Frame:", len(frame_pcds))
    R, t, _ = icp(target, frame_pcds[0], mode='kd_tree', epsilon=epsilon, max_iters=max_iters)
    trans = R @ source + t
    merged_pcd = R @ merged_pcd + t
    merged_pcd = np.hstack((merged_pcd, trans))

    return merged_pcd


if __name__ == "__main__":
    
    #3.1 Results
    N = 1 #Choose a value between {1, 2, 4, 10}
    file_path = f'./Data/merge_end/N{N}.npy'
    merged_pcd = np.load(file_path)
    visualize(merged_pcd)

    #3.2 Results
    # Uncomment to get 3.2 results
    # N = 10 #Choose a value between {4, 10}
    # file_path = f'./Data/merge_between/N{N}.npy'
    # merged_pcd = np.load(file_path)
    # visualize(merged_pcd)
    
