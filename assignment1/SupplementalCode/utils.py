import os
import numpy as np
import open3d as o3d
from random import sample

# globals.
DATA_DIR = 'Data'
FRAMES_PATH = DATA_DIR + '/data/'

# == Load data ==
def get_cleaned_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path)
    
    # convert into ndarray
    pcd_arr = np.asarray(pcd.points)

    # clean the point cloud using a threshold
    dist = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[dist < 1]

    return pcd_arr_cleaned

def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy'))
    return source, target


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy'))
    return source, target


def subsample_graph(A1, points=10000):
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


def plot_progress(source, target, trans, iter=0, dir='./figures/waves', save_figure=True):
    
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

    if save_figure:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(source_pcd)
        vis.add_geometry(target_pcd)
        vis.add_geometry(trans_pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(dir + f'/prog_{iter}.png')


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