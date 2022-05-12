from fundamental_matrix import *
import cv2 as cv
import numpy as np
from scipy import spatial


def shared_points(mtx):
    '''
    Obtain the dense Point-view matrix.
    '''
    return mtx[:, np.all(mtx != -1.0, axis=0)]


def construct_new_pv_row(X1, Y1, X2, Y2, cur_x, cur_y):
    '''
    Find matches between Point-view matrix rows and add new column when
    new point found.
    '''
    X2_new = np.full(len(cur_x), -1.0)
    Y2_new = np.full(len(cur_y), -1.0)

    for j in range(len(X1)):
        pointx = X1[j]
        pointy = Y1[j]
        found = False

        # Find a matching point.
        for m in range(len(cur_x)):
            if cur_x[m] != -1 and cur_y[m] != -1:

                # If we find a point, add it and break.
                if pointx == cur_x[m] and pointy == cur_y[m]:
                    X2_new[m] = X2[j]
                    Y2_new[m] = Y2[j]
                    found = True
                    break

        # Add a new column if no new points found.
        if not found:
            arr_len = len(cur_x)

            X2_new = np.insert(X2_new, arr_len, X2[j])
            Y2_new = np.insert(Y2_new, arr_len, Y2[j])

            cur_x = np.insert(cur_x, arr_len, X1[j])
            cur_y = np.insert(cur_y, arr_len, Y1[j])

    return cur_x, cur_y, X2_new, Y2_new


def update_columns(pv_mtx, new_cols):
    '''
    Update rows of the Point-view matrix where no matching points
    were found.
    '''
    zeroes = np.full(new_cols, -1.0)
    new_pv_mtx = []
    for row in pv_mtx:
        new_row = np.append(row, zeroes)
        new_pv_mtx.append(new_row)

    return new_pv_mtx


def create_pv_matrix(start=1, N=50, dist_ratio=0.7):
    '''
    Create Point-view matrix with images start to N.
    '''
    pv_mtx = []

    img1 = cv.imread(f'./Data/house/frame000000{"{0:02}".format(start)}.png')
    img2 = cv.imread(f'./Data/house/frame000000{"{0:02}".format(start+1)}.png')

    points1, points2 = calculate_keypoint_matching(img1, img2, dist_ratio, draw=False)

    # pv_mtx.append(points1[:, 0])
    # pv_mtx.append(points1[:, 1])
    pv_mtx.append(points2[:, 0])
    pv_mtx.append(points2[:, 1])

    for i in range(start + 1, N):
        img1 = cv.imread(f'./Data/house/frame000000{"{0:02}".format(i)}.png')
        if i == N-1:
            img2 = cv.imread(f'./Data/house/frame000000{"{0:02}".format(start)}.png')
        else:
            img2 = cv.imread(f'./Data/house/frame000000{"{0:02}".format(i+1)}.png')
        
        # Keypoint matching
        points1, points2 = calculate_keypoint_matching(img1, img2, dist_ratio, draw=False)
        if len(points1) == 0:
            continue

        # Obtain the row with the already obtained matches.
        cur_x = pv_mtx[2*(i-2)]
        cur_y = pv_mtx[2*(i-2) + 1]

        # Obtain the rows of the new matches.
        X1 = points1[:, 0]
        Y1 = points1[:, 1]
        X2 = points2[:, 0]
        Y2 = points2[:, 1]

        # Obtain the new rows.
        new_x, new_y, new_X2, new_Y2 = construct_new_pv_row(X1, Y1, X2, Y2, cur_x, cur_y)

        # If new columns were added update the Point-view matrix.
        if len(new_x) != len(cur_x):
            pv_mtx = update_columns(pv_mtx[:-2], len(new_x) - len(cur_x))
            pv_mtx.append(new_x)
            pv_mtx.append(new_y)
        pv_mtx.append(new_X2)
        pv_mtx.append(new_Y2)

    return np.array(pv_mtx)


def visualize_pvm(pvm):
    pvm[pvm!=-1] = 0
    pvm[pvm==-1] = 1
    plt.imshow(pvm, cmap='gray', aspect=20)
    plt.show()


if __name__ == '__main__':

    # pvm = np.loadtxt('PVM_ours_last.txt')
    # pvm = np.loadtxt('PointViewMatrix.txt')
    pvm = create_pv_matrix()
    # np.savetxt('PVM_ours_last.txt', pvm)
    print(pvm.shape)
    visualize_pvm(pvm.copy())
