import numpy as np
import open3d as o3d
from utils import *
from icp import *

############################
#   Merge Scene            #
############################

def estimate_transformations(frame_pcds, epsilon, max_iters, kd_tree):
    
    #  Estimate the camera poses using two consecutive frames of given data.
    rotations = []
    translations = []
    for i in range(len(frame_pcds)-1):
        print('Frame ', i+1)
        R, t = icp(frame_pcds[i], frame_pcds[i+1], kd_tree=kd_tree, epsilon=epsilon, max_iters=max_iters)
        rotations.append(R)
        translations.append(t)
    
    return rotations, translations


def merge_scene_in_between(frame_pcds, epsilon=1e-7, max_iters=100, visualize=True):
    '''
    Estimates the camera poses and merges every N frames
    '''

    #  Iteratively merge and estimate the camera poses for the consecutive frames.
    source_frame = frame_pcds[0]
    for i, frame in enumerate(frame_pcds[1:]):
        print("Frame:", i+1)
        R, t = icp(source_frame, frame, sampling='none', ratio=0.1, kd_tree=True, epsilon=epsilon, max_iters=max_iters)
        new_source = R @ source_frame + t
        source_frame = np.hstack((new_source, frame))
        
        if(i%1 == 0):
            trans_pcd = o3d.geometry.PointCloud()
            trans_pcd.points = o3d.utility.Vector3dVector(source_frame.T)
            o3d.visualization.draw_geometries([trans_pcd])
    
    trans = subsample_graph(source_frame, points=20000)
    trans_pcd = o3d.geometry.PointCloud()
    trans_pcd.points = o3d.utility.Vector3dVector(trans.T)

    if visualize:
        o3d.visualization.draw_geometries([trans_pcd])


def merge_scene_end(frame_pcds, epsilon=1e-7, max_iters=100, visualize=True):
    '''
    Estimates the camera poses of every N frames and merges the results at the end
    '''

    # Estimate the camera poses for the consecutive frames and merge at the end
    # rotations, translations = estimate_transformations(frame_pcds, epsilon=1e-7, max_iters=50, kd_tree=True)

    for i in range(len(frame_pcds)-1):
        print("Frame:", i+1)
        source = frame_pcds[i]
        target = frame_pcds[i+1]
        R, t = icp(source, target, kd_tree=True, epsilon=epsilon, max_iters=max_iters)
        trans = R @ source + t
        if i == 0:
            merged_pcd = trans
        else:
            merged_pcd = R @ merged_pcd + t
            merged_pcd = np.hstack((merged_pcd, trans))
        
        if(i%10 == 0):
            trans_pcd = o3d.geometry.PointCloud()
            trans_pcd.points = o3d.utility.Vector3dVector(merged_pcd.T)
            o3d.visualization.draw_geometries([trans_pcd])
        
    if visualize:
        o3d.visualization.draw_geometries([trans_pcd])

if __name__ == "__main__":

    frame_pcds = get_frame_pointclouds(step=4, frame_count=100) # step = N = {1, 2, 4, 10}
    merge_scene_end(frame_pcds, epsilon=1e-7, max_iters=100)
    merge_scene_in_between(frame_pcds, epsilon=1e-7, max_iters=100)
