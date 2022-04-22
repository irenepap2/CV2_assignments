import os
import numpy as np
import open3d as o3d
from random import sample

# globals.
DATA_DIR = '../Data'
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
    save_dir = './figures/waves'
    return source, target, save_dir


def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy'))
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy'))
    save_dir = './figures/bunnies'
    return source, target, save_dir


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


def obtain_informative_regions(A1, A2, alpha, total_p):
    '''
    Obtain informative region of mesh and return new array of points.
    '''
    # obtain pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(A1.T)

    # create mesh from pointcloud
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()

    # set the number of points to sample
    if A1.shape[1] > 2000:
        num_p = A1.shape[1] // 2000 * 2000
    else:
        num_p = A1.shape[1] // 200 * 200

    # sample points and create line set
    pcl = mesh.sample_points_poisson_disk(number_of_points=num_p)
    hull, _ = pcl.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)

    # extract points from line set and create new A1 and A2
    ls_points = np.asarray(hull_ls.points)
    new_A1 = informative_region(ls_points)
    new_A2 = subsample_graph(A2, points=total_p)

    return np.array(new_A1), np.array(new_A2)


def informative_region(points):
    '''
    Create new A1 based of calculated informative points
    '''
    new_A1 = [[], [], []]
    for point in points:
        new_A1[0].append(point[0])
        new_A1[1].append(point[1])
        new_A1[2].append(point[2])
    return new_A1


def set_multi_res(A1, epsilon, N):
    '''
    Set the sample points and create resolution threshold.
    '''
    total_p = A1.shape[1]
    cur_eps = epsilon
    points_sampled = []
    steps = []
    while total_p > 100:
        points_sampled.insert(0, total_p)
        total_p = total_p // N
        cur_eps *= 2*N
        steps.insert(0, cur_eps)
    return points_sampled, steps


def gauss_noise(A1, ratio=0.1):
    '''
    Add a ratio of noise to the image.
    '''
    noise_len = int(ratio*len(A1[0]))
    A1_X = np.random.normal(np.mean(A1[0]) * 0.05, np.std(A1[0]) * 0.05, (noise_len, 1))
    A1_Y = np.random.normal(np.mean(A1[1]) * 0.05, np.std(A1[1]) * 0.05, (noise_len, 1))
    A1_Z = np.random.normal(np.mean(A1[2]) * 0.05, np.std(A1[2]) * 0.05, (noise_len, 1))
    gen_points = sample(range(0, len(A1[0] - 1)), noise_len)
    for i, point in enumerate(gen_points):
        A1[0][point] += A1_X[i]
        A1[1][point] += A1_Y[i]
        A1[2][point] += A1_Z[i]

    return A1


def plot_progress(source, target, trans, iter=0, dir='./figures/waves', save_figure=True, plot_source=True):

    # visualization from ndarray
    if plot_source:
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source.T)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target.T)
    trans_pcd = o3d.geometry.PointCloud()
    trans_pcd.points = o3d.utility.Vector3dVector(trans.T)
    if plot_source:
        source_pcd.paint_uniform_color([1, 0, 0])
    target_pcd.paint_uniform_color([0, 1, 0])
    trans_pcd.paint_uniform_color([0, 0, 1])
    if plot_source:
        o3d.visualization.draw_geometries([source_pcd, target_pcd, trans_pcd])
    else:
        o3d.visualization.draw_geometries([target_pcd, trans_pcd])

    if save_figure:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        if plot_source:
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
