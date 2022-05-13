import matplotlib.pyplot as plt
import random
import cv2 as cv
import numpy as np
from scipy.linalg import orthogonal_procrustes

def plot_epipolar_lines(img1, img2, points1, points2, F_matrix):

    # compute epipolar lines corresponding to points in image 2
    lines1 = points2 @ F_matrix

    # compute epipolar lines corresponding to points in image 1
    lines2 = points1 @ F_matrix

    _, w, _ = img1.shape
    # draw epipolar lines on image 1 corresponding to points in image 2
    for i, l in enumerate(lines1):
        color = (random.randint(128,255), random.randint(128,255), random.randint(128,255))
        pt1 = (0, int(-l[2]/l[1]))
        pt2 = (w, int((-l[2]-w*l[0])/l[1]))
        img1 = cv.line(img1, pt1, pt2, color)
        img2 = cv.circle(img2, (int(points2[i, 0]), int(points2[i, 1])), 3, color, 3)

    # # draw epipolar lines on image 2 corresponding to points in image 1
    for i, l in enumerate(lines2):
        color = (random.randint(128,255), random.randint(128,255), random.randint(128,255))
        pt1 = (0, int(-l[2]/l[1]))
        pt2 = (w, int((-l[2]-w*l[0])/l[1]))
        img2 = cv.line(img2, pt1, pt2, color)
        img1 = cv.circle(img1, (int(points1[i, 0]), int(points1[i, 1])), 3, color, 3)

    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()


def drawlines(img1,img2,lines,pts1,pts2):

    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def procrustes(data1, data2):
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))
    
    return mtx1, mtx2, disparity, R, s, norm1, norm2