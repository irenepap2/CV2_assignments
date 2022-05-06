import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random
from utils import *


def calculate_keypoint_matching(img1, img2, dist_ratio, draw=True):
    '''
    Finds the keypoints between the two images img1 and img2
    and their corresponding matches
    SIFT descriptors to find the keypoints
    BFMatcher to match descriptors
    '''
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None) # queryImage
    kp2, des2 = sift.detectAndCompute(img2, None) # trainImage

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # ratio test as per Lowe's paper
    good = []
    points1 = []
    points2 = []
    for m, n in matches:
        if m.distance < dist_ratio*n.distance:
            good.append([m])
            points1.append(kp1[m.queryIdx].pt) #append kp1 coordinates (index of the descriptor in query descriptors)
            points2.append(kp2[m.trainIdx].pt) #append kp2 coordinates (index of the descriptor in train descriptors)
            
    
    # cv.drawMatchesKnn expects list of lists as matches.
    if draw:
        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()

    points1 = np.array(points1)
    points2 = np.array(points2)

    # convert points to homogeneous coordinates
    points1 = np.concatenate((points1, np.ones((len(points1),1))), axis=1)
    points2 = np.concatenate((points2, np.ones((len(points2),1))), axis=1)
    
    return points1, points2


def normalize_points(points):
    '''
    Normalizes points so that their mean is 0 and average distance to the mean is sqrt(2)
    '''

    x = points[:,0]
    y = points[:,1]
    m_x = np.mean(x)
    m_y = np.mean(y)
    d = np.mean(np.sqrt((x - m_x)**2 + (y - m_y)**2))

    T = np.array([[np.sqrt(2)/d, 0, -m_x*np.sqrt(2)/d], 
                  [0, np.sqrt(2)/d, -m_y*np.sqrt(2)/d], 
                  [0, 0, 1]])

    points_normalized = T @ points.T

    # print('Distance:', np.mean([np.linalg.norm(x) for x in points_normalized.T[:, :2]]))
    
    return points_normalized.T, T


def findFundamentalMatrixOpenCV(points1, points2):
    '''
    Computes Fundamental Matrix using the opencv function
    '''

    F, _ = cv.findFundamentalMat(points1, points2)
    return F


def findFundamentalMatrixEightPointAlgo(points1, points2, normalize=False):
    '''
    Computes Fundamental Matrix using the EPA algorithm (3.1)
    If normalize=True, it normalizes points1 and points2 and then computes the Fundamental Matrix (3.2)
    '''

    if normalize:
        points1, T1 = normalize_points(points1)
        points2, T2 = normalize_points(points2)
    
    # print('Distance of points1:', np.mean([np.linalg.norm(x) for x in points1[:, :2]]))
    # print('Distance of points2:', np.mean([np.linalg.norm(x) for x in points2[:, :2]]))
    
    # construct the nx9 matrix A
    A = np.array([points1[:,0] * points2[:,0], 
                  points1[:,0] * points2[:,1], 
                  points1[:,0], 
                  points1[:,1] * points2[:,0],
                  points1[:,1] * points2[:,1], 
                  points1[:,1], 
                  points2[:,0], 
                  points2[:,1], 
                  np.ones((len(points1)))]).T

    _, _, VT = np.linalg.svd(A)
    # The entries of F are the components of the column of V corresponding to the smallest singular value (last value in D)
    F = VT.T[:,-1].reshape(3,3)

    # Find the SVD of F
    Uf, Df, VfT = np.linalg.svd(F)
    # Set the smallest singular value in the diagonal matrix Df to zero
    Df[-1] = 0
    # Recompute F
    F = Uf @ np.diag(Df) @ VfT

    if normalize:
        F = T2.T @ F @ T1

    # print(np.mean(A @ F.reshape(9,1)))

    return F


def find_inliers(F, points1, points2, sampson_thres=1):
    points1_inliers = []
    points2_inliers = []
    for i in range(len(points1)):
        d = (points1[i].T @ F @ points2[i]) ** 2 / ((F @ points1[i])[0] ** 2 + (F @ points1[i])[1] ** 2 + (F @ points2[i])[0] ** 2 + (F @ points2[i])[1] ** 2)
        if d < sampson_thres:
            points1_inliers.append(points1[i])
            points2_inliers.append(points2[i])

    return np.array(points1_inliers), np.array(points2_inliers)


def findFundamentalMatrixRansac(points1, points2, num_iterations):

    max_inliers = 0
    max_points1_inliers = []
    max_points2_inliers = []
    for i in range(num_iterations):
        indices = [ind for ind,_ in random.sample(list(enumerate(points1)), 8)]
        points1_8 = points1[indices]
        points2_8 = points2[indices]

        F_matrix = findFundamentalMatrixEightPointAlgo(points1_8, points2_8, normalize=True)
        points1_inliers, points2_inliers = find_inliers(F_matrix, points1, points2)
        assert len(points1_inliers) == len(points2_inliers)

        if len(points1_inliers) > max_inliers:
            max_inliers = len(points1_inliers)
            max_points1_inliers = points1_inliers
            max_points2_inliers = points2_inliers

    F_matrix = findFundamentalMatrixEightPointAlgo(max_points1_inliers, max_points2_inliers)
    return F_matrix, max_inliers
    

if __name__ == '__main__':
    
    img1_num = 1
    img2_num = 2
    dist_ratio = 0.7
    random.seed(10)

    img1 = cv.imread(f'./Data/house/frame0000000{img1_num}.png') # queryImage
    img2 = cv.imread(f'./Data/house/frame0000000{img2_num}.png') # trainImage

    counts = ["{0:02}".format(i) for i in range(1, 50)]

    #load images
    # for i in range(len(counts)-1):
    #     img1 = cv.imread(f'./Data/house/frame000000{counts[i]}.png') # queryImage
    #     img2 = cv.imread(f'./Data/house/frame000000{counts[i+1]}.png') # trainImage
    
    #calculate matching keypoints
    points1, points2 = calculate_keypoint_matching(img1, img2, dist_ratio)

    # F_matrix = findFundamentalMatrixOpenCV(points1, points2)
    # plot_epipolar_lines(img1.copy(), img2.copy(), points2, F_matrix)
    # F_matrix = findFundamentalMatrixEightPointAlgo(points1.copy(), points2.copy(), normalize=True)
    F_matrix, max_inliers = findFundamentalMatrixRansac(points1.copy(), points2.copy(), num_iterations=200)
    print(max_inliers)
    # F_matrix = findFundamentalMatrixEightPointAlgoRansac(points1.copy(), points2.copy())
    plot_epipolar_lines(img1.copy(), img2.copy(), points2, F_matrix)
    # plot_epipolar_lines(img2.copy(), img1.copy(), points1, F_matrix)

