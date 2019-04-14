import numpy as np
from numpy.linalg import inv
import cv2
import glob
from PIL import Image
#from sympy import *
from scipy.optimize import fsolve, leastsq

def func(i, h_vec):
	x, y, z ,a, b ,c= i[0], i[1], i[2], i[3], i[4], i[5]
	t = []
	for i in range(7):
		t.append(h_vec[i][0]*x + h_vec[i][1]*y + h_vec[i][2]*z + h_vec[i][3]*a + h_vec[i][4]*b + h_vec[i][5]*c)
	return t
	# return[
	# 		h_vec[0][0]*x + h_vec[0][1]*y + h_vec[0][2]*z + h_vec[0][3]*a + h_vec[0][4]*b + h_vec[0][5]*c,   
	# 		h_vec[1][0]*x + h_vec[1][1]*y + h_vec[1][2]*z + h_vec[1][3]*a + h_vec[1][4]*b + h_vec[1][5]*c,
	# 		h_vec[2][0]*x + h_vec[2][1]*y + h_vec[2][2]*z + h_vec[2][3]*a + h_vec[2][4]*b + h_vec[2][5]*c,
	# 		h_vec[3][0]*x + h_vec[3][1]*y + h_vec[3][2]*z + h_vec[3][3]*a + h_vec[3][4]*b + h_vec[3][5]*c,
	# 		h_vec[4][0]*x + h_vec[4][1]*y + h_vec[4][2]*z + h_vec[4][3]*a + h_vec[4][4]*b + h_vec[4][5]*c,
	# 		h_vec[5][0]*x + h_vec[5][1]*y + h_vec[5][2]*z + h_vec[5][3]*a + h_vec[5][4]*b + h_vec[5][5]*c,

	# ]

def get_extrinsic_matrix(objpoints, imgpoints, img_size):

	img_pts=[]
	obj_pts=[]
	H=[]

	for idx in range(10):

		C = []
		for i in range(49):
			c = np.array([0, 0, 0, -1 * objpoints[idx][i][0], -1 * objpoints[idx][i][1], -1,
						  imgpoints[idx][i][0][1] * objpoints[idx][i][0],
						  imgpoints[idx][i][0][1] * objpoints[idx][i][1], imgpoints[idx][i][0][1]])
			C.append(c)
			c = np.array([objpoints[idx][i][0], objpoints[idx][i][1], 1, 0, 0, 0,
						  -1 * imgpoints[idx][i][0][0] * objpoints[idx][i][0],
						  -1 * imgpoints[idx][i][0][0] * objpoints[idx][i][1], -1 * imgpoints[idx][i][0][0]])
			C.append(c)
		C = np.array(C)

		cu, cd, cv = np.linalg.svd(C)
		cv = np.transpose(cv)
		hh = cv[:, 8]



	for i in range(len(imgpoints)):

		obj_grid = objpoints[i]
		img_grid = imgpoints[i]

		num_grids = obj_grid.shape[0]
		obj_grid = np.array(obj_grid[:, np.newaxis, :])
		# Pad obj_grid with [1,...,1] for z-axis: [U,V,1]
		obj_grid[:, :, 2] = 1
		img_grid = np.concatenate([img_grid, np.ones([num_grids, 1, 1])], axis=2)
		homography = cv2.findHomography(obj_grid, img_grid, cv2.RANSAC, 5.0)[0]


		img_pts.append(np.reshape(imgpoints[i], (len(imgpoints[i]),2)))

		img_pts[i] = np.insert(img_pts[i], 2, float(1.0), axis=1)

		obj_pts.append(objpoints[i, np.newaxis, :])

		for j in range(objpoints[i].shape[0]):
			obj_pts[i][0][j][2] = float(1.0)


		obj_pts_array = np.array(obj_pts[i])
		img_pts_array = np.array(img_pts[i])

		homography_ = cv2.findHomography(obj_pts_array, img_pts_array, cv2.RANSAC, 5.0)[0]

		print(homography)
		print("=============")
		print(homography_)

		# obj_pts_trans = np.transpose(obj_pts[i])
		#
		# mul_ob  = np.dot(obj_pts_trans , obj_pts[i])
		# mul_img = np.dot(np.transpose(img_pts[i]) , obj_pts[i])
		#
		# H.append(np.dot(mul_img , inv(mul_ob)))

		H.append(homography_)


		

	h_vec = np.zeros((7,6))

	for k in range(7):
		h_vec[k][0] = H[k][0][0] * H[k][1][0] 						    #h12*h21
		h_vec[k][1] = H[k][0][0] * H[k][1][1] + H[k][0][1] * H[k][1][0] #h11*h22+h12*h21
		h_vec[k][2] = H[k][0][1] * H[k][1][1]							#h12*h22
		h_vec[k][3] = H[k][0][2] * H[k][1][0] + H[k][0][0] * H[k][1][2] #h13*h21+h11*h23
		h_vec[k][4] = H[k][0][2] * H[k][1][1] + H[k][0][1] * H[k][1][2] #h13*h22+h12*h23
		h_vec[k][5] = H[k][0][2] * H[k][1][2] 							#h13*h23


	zero_vec = np.zeros((3,1))

	#b = np.linalg.svd(h_vec)[2]
	#b_transpose = np.transpose(b)

	u, s, vh = np.linalg.svd(v, full_matrices=False)

	#b = leastsq(func, [0,0,0,0,0,0], h_vec, full_output=1)
	# b = nonlinsolve([2,3,4,5,3,4], [0,0,0,0,0,0])

	# for i in range(len(b[1])):
	# 	print(np.dot(h_vec, b[1][i]))

	b = b[1][5]

	ov = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] * b[1])
	lamda = b[5] - (b[3] * b[3] + ov * (b[1]*b[3] - b[0] * b[4])) / b[0]
	fu = pow((lamda/b[0]),0.5)
	fv = pow((lamda * b[0] / (b[0] * b[2] - b[1] * b[1])),0.5)

	gamma = -b[1] * fu * fu * fv / lamda
	ou = lamda * ov / fv - b[3] * fu * fu / lamda
	ou = - b[3] * fu * fu / lamda

	A = np.ndarray(shape=(3,3), dtype=float)
	A[0][0] = fu
	A[0][1] = 0
	A[0][2] = ou
	A[1][0] = 0
	A[1][1] = fv
	A[1][2] = ov
	A[2][0] = 0
	A[2][1] = 0
	A[2][2] = 1

	print("Intrinsic vector" , A)

	# for i in range(len(h_vec)):
	# 	h_t = np.transpose(h_vec[i]),
	# 	r1 = lamda * np.dot(np.linalg.inv(A),h_t[0])
	# 	r2 = lamda * np.dot(np.linalg.inv(A),h_t[1])
	# 	t = lamda * np.dot(np.linalg.inv(A),h_t[2])

	# print(H[2])
	# a = np.dot(H[2],np.transpose(obj_pts[2]))

	# print(np.transpose(a))
	# print("aaaaa")
	# print(img_pts[2])

if __name__ == "__main__":
	a = np.load(open("img.npy", "rb"))
	b = np.load(open("obj.npy", "rb"))

	get_extrinsic_matrix(b, a, 2)

	