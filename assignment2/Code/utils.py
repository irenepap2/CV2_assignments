import matplotlib.pyplot as plt
import random
import cv2 as cv

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