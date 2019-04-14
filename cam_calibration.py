import numpy as np

from numpy.linalg import inv

from scipy.optimize import leastsq


def func(i, h_vec):
    x, y, z, a, b, c = i[0], i[1], i[2], i[3], i[4], i[5]

    t = []

    for i in range(7):
        t.append(h_vec[i][0] * x + h_vec[i][1] * y + h_vec[i][2] * z + \
                 h_vec[i][3] * a + h_vec[i][4] * b + h_vec[i][5] * c)

    return t

def get_intrinsic_matrix(objpoints, imgpoints):

    img_pts = []
    obj_pts = []
    H = []

    for i in range(len(imgpoints)):

        img_pts.append(np.reshape(imgpoints[i], (len(imgpoints[i]), 2)))
        img_pts[i] = np.insert(img_pts[i], 2, float(1.0), axis=1)

        obj_pts.append(objpoints[i])

        for j in range(objpoints[i].shape[0]):
            obj_pts[i][j][2] = float(1.0)

        obj_pts_trans = np.transpose(obj_pts[i])

        mul_ob  = np.dot(obj_pts_trans, obj_pts[i])
        mul_img = np.dot(np.transpose(img_pts[i]), obj_pts[i])

        H.append(np.dot(mul_img, inv(mul_ob)))

    h_vec = np.zeros((7, 6))

    for k in range(7):
        h_vec[k][0] = H[k][0][0] * H[k][1][0]  # h12*h21
        h_vec[k][1] = H[k][0][0] * H[k][1][1] + H[k][0][1] * H[k][1][0]  # h11*h22+h12*h21
        h_vec[k][2] = H[k][0][1] * H[k][1][1]  # h12*h22
        h_vec[k][3] = H[k][0][2] * H[k][1][0] + H[k][0][0] * H[k][1][2]  # h13*h21+h11*h23
        h_vec[k][4] = H[k][0][2] * H[k][1][1] + H[k][0][1] * H[k][1][2]  # h13*h22+h12*h23
        h_vec[k][5] = H[k][0][2] * H[k][1][2]  # h13*h23

    # ==================================Get K matrix=========================================

    b = leastsq(func, [0, 0, 0, 0, 0, 0], h_vec, full_output=1)
    b = b[1][5]

    ov = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] * b[1])
    lamda = b[5] - (b[3] * b[3] + ov * (b[1] * b[3] - b[0] * b[4])) / b[0]
    fu = pow((lamda / b[0]), 0.5)
    fv = pow((lamda * b[0] / (b[0] * b[2] - b[1] * b[1])), 0.5)

    gamma = -b[1] * fu * fu * fv / lamda
    ou = lamda * ov / fv - b[3] * fu * fu / lamda
    ou = - b[3] * fu * fu / lamda

    # ===============================Get Intrisic matrix=====================================    
    
    intrisic_mtx = np.ndarray(shape=(3, 3), dtype=float)
    intrisic_mtx[0][0] = fu
    intrisic_mtx[0][1] = 0
    intrisic_mtx[0][2] = ou
    intrisic_mtx[1][0] = 0
    intrisic_mtx[1][1] = fv
    intrisic_mtx[1][2] = ov
    intrisic_mtx[2][0] = 0
    intrisic_mtx[2][1] = 0
    intrisic_mtx[2][2] = 1

    return intrisic_mtx


if __name__ == "__main__":
    a = np.load(open("img.npy", "rb"))
    b = np.load(open("obj.npy", "rb"))

    get_intrinsic_matrix(b, a)

