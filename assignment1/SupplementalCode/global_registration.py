import numpy as np
import open3d as o3d
from utils import *
from example import *

############################
#   Merge Scene            #
############################

def estimate_transformations(frame_pcds):
    
    #  Estimate the camera poses using two consecutive frames of given data.
    rotations = []
    translations = []
    for i in range(len(frame_pcds)-1):
        print('Frame ', i+1)
        R, t = icp(frame_pcds[i], frame_pcds[i+1], kd_tree=True, epsilon=1e-4, max_iters=50)
        rotations.append(R)
        translations.append(t)
    
    return rotations, translations


def merge_scene(frame_pcds, merge_in_between=False, visualize=True):
    '''
    Estimates the camera poses of every N frames and merges the results
    (either in between every ipc calculation, or all together at the end)
    '''

    #  Iteratively merge and estimate the camera poses for the consecutive frames.
    if merge_in_between:
        source_frame = frame_pcds[0]
        for i, frame in enumerate(frame_pcds[1:]):
            print("Frame:", i+1)
            R, t = icp(source_frame, frame, kd_tree=True, epsilon=1e-2, max_iters=10)
            new_source = R @ source_frame + t
            source_frame = np.hstack((new_source, frame))
            
            if(i%5 == 0):
                trans_pcd = o3d.geometry.PointCloud()
                trans_pcd.points = o3d.utility.Vector3dVector(source_frame.T)
                o3d.visualization.draw_geometries([trans_pcd])
        
        trans = subsample_graph(source_frame, points=20000)
        trans_pcd = o3d.geometry.PointCloud()
        trans_pcd.points = o3d.utility.Vector3dVector(trans.T)
        trans_pcds = [trans_pcd]

    # Estimate the camera poses for the consecutive frames and merge at the end
    else:
        rotations, translations = estimate_transformations(frame_pcds)
        
        trans_pcds = []
        for i, frame in enumerate(frame_pcds):
            print('Frame', i)
            r_list = rotations[i:]
            t_list = translations[i:]
            for r, t in zip(r_list, t_list):
                frame = r @ frame + t

            # trans = subsample_graph(frame)
            trans = frame
            trans_pcd = o3d.geometry.PointCloud()
            trans_pcd.points = o3d.utility.Vector3dVector(trans.T)
            trans_pcds.append(trans_pcd)

            if(i%5 == 0):
                o3d.visualization.draw_geometries(trans_pcds)

    if visualize:
        o3d.visualization.draw_geometries(trans_pcds)

if __name__ == "__main__":

    frame_pcds = get_frame_pointclouds(step=1, frame_count=100) # step = N = {1, 2, 4, 10}
    merge_scene(frame_pcds, merge_in_between=True)