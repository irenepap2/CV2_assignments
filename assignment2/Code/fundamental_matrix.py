import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import random


def calculate_keypoint_matching(img1, img2, dist_ratio):
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
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()

    points1 = np.array(points1)
    points2 = np.array(points2)

    # convert points to homogeneous coordinates
    points1 = np.concatenate((points1, np.ones((len(points1),1))), axis=1)
    points2 = np.concatenate((points2, np.ones((len(points2),1))), axis=1)
    
    return points1, points2


def findFundamentalMatrixOpenCV(points1, points2):
    F, _ = cv.findFundamentalMat(points1, points2)
    return F


def findFundamentalMatrixEightPointAlgo(points1, points2):
    # construct the nx9 matrix A
    A = np.array([points1[:,0] * points2[:,0], 
                  points1[:,0] * points2[:,1], 
                  points1[:,0], 
                  points1[:,1] * points2[:,0],
                  points1[:,1] * points2[:,1], 
                  points1[:,1], 
                  points2[:,0], 
                  points2[:,1], 
                  np.ones((len(points1)))])

    U, D, Vt = np.linalg.svd(A)
    V = Vt.T


def plot_epipolar_lines(img1, img2, points, F_matrix):

    # compute epipolar lines corresponding to points in image 2
    lines1 = points @ F_matrix

    _, w, _ = img1.shape
    # draw epipolar lines on image 1 corresponding to points in image 2
    for i, l in enumerate(lines1):
        color = (random.randint(128,255), random.randint(128,255), random.randint(128,255))
        pt1 = (0, int(-l[2]/l[1]))
        pt2 = (w, int((-l[2]-w*l[0])/l[1]))
        img1_lines = cv.line(img1, pt1, pt2, color)
        img2_points = cv.circle(img2, (int(points[i, 0]), int(points[i, 1])), 3, color, 1)

    plt.subplot(121)
    plt.imshow(img1_lines)
    plt.subplot(122)
    plt.imshow(img2_points)
    plt.show()


if __name__ == '__main__':
    
    img1_num = 1
    img2_num = 2
    dist_ratio = 0.2

    #load images
    img1 = cv.imread(f'./Data/house/frame0000000{img1_num}.png') # queryImage
    img2 = cv.imread(f'./Data/house/frame0000000{img2_num}.png') # trainImage
    
    #calculate matching keypoints
    points1, points2 = calculate_keypoint_matching(img1, img2, dist_ratio)
    F_matrix = findFundamentalMatrixOpenCV(points1, points2)
    plot_epipolar_lines(img1.copy(), img2.copy(), points2, F_matrix)
    plot_epipolar_lines(img2.copy(), img1.copy(), points1, F_matrix)

