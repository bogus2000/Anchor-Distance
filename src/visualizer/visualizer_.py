import numpy as np
import cv2
import time
import scipy

Tr_velo_to_cam = np.array(
    [[7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04, -4.069766000000e-03],
     [1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02],
     [9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02, -2.717806000000e-01],
     [0, 0, 0, 1]])
Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)

caliv = np.array(
    [[9.786977e+02, 0.000000e+00, 6.900000e+02, 0],
     [0.000000e+00, 9.717435e+02, 2.497222e+02, 0],
     [0.000000e+00, 0.000000e+00, 1.000000e+00, 0],
     [0, 0, 0, 0]])
trans = np.array(
    [[0, 0, 0, 0],
     [0, 0, 0, 0.0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
)
caliv_trans = np.matmul(caliv, trans)


P0 = np.array(
    [[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 0.000000000000e+00],
     [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00],
     [0, 0, 0, 1]])
P1 = np.array(
    [[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.875744000000e+02],
     [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00],
     [0, 0, 0, 1]])
P2 = np.array(
    [[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
     [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03],
     [0, 0, 0, 1]])
P3 = np.array(
    [[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.395242000000e+02],
     [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.199936000000e+00],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.729905000000e-03],
     [0, 0, 0, 1]])

kitti_proj_mat = P0 + caliv_trans
kitti_proj_mat_inv = np.linalg.inv(kitti_proj_mat)

offset = np.zeros((64,64,64,3))
for i in range(64):
    for j in range(64):
        for k in range(64):
            offset[i,j,k,:] = i,j,k


def matmul3x3(a, b):
    c00 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0]
    c01 = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1]
    c02 = a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2]

    c10 = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0]
    c11 = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1]
    c12 = a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2]

    c20 = a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0]
    c21 = a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1]
    c22 = a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2]

    return np.array([[c00, c01, c02],
                     [c10, c11, c12],
                     [c20, c21, c22]])


def matmul4x4(a, b):
    c00 = a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] + a[0, 2] * b[2, 0] + a[0, 3] * b[3, 0]
    c01 = a[0, 0] * b[0, 1] + a[0, 1] * b[1, 1] + a[0, 2] * b[2, 1] + a[0, 3] * b[3, 1]
    c02 = a[0, 0] * b[0, 2] + a[0, 1] * b[1, 2] + a[0, 2] * b[2, 2] + a[0, 3] * b[3, 2]
    c03 = a[0, 0] * b[0, 3] + a[0, 1] * b[1, 3] + a[0, 2] * b[2, 3] + a[0, 3] * b[3, 3]

    c10 = a[1, 0] * b[0, 0] + a[1, 1] * b[1, 0] + a[1, 2] * b[2, 0] + a[1, 3] * b[3, 0]
    c11 = a[1, 0] * b[0, 1] + a[1, 1] * b[1, 1] + a[1, 2] * b[2, 1] + a[1, 3] * b[3, 1]
    c12 = a[1, 0] * b[0, 2] + a[1, 1] * b[1, 2] + a[1, 2] * b[2, 2] + a[1, 3] * b[3, 2]
    c13 = a[1, 0] * b[0, 3] + a[1, 1] * b[1, 3] + a[1, 2] * b[2, 3] + a[1, 3] * b[3, 3]

    c20 = a[2, 0] * b[0, 0] + a[2, 1] * b[1, 0] + a[2, 2] * b[2, 0] + a[2, 3] * b[3, 0]
    c21 = a[2, 0] * b[0, 1] + a[2, 1] * b[1, 1] + a[2, 2] * b[2, 1] + a[2, 3] * b[3, 1]
    c22 = a[2, 0] * b[0, 2] + a[2, 1] * b[1, 2] + a[2, 2] * b[2, 2] + a[2, 3] * b[3, 2]
    c23 = a[2, 0] * b[0, 3] + a[2, 1] * b[1, 3] + a[2, 2] * b[2, 3] + a[2, 3] * b[3, 3]

    c30 = a[3, 0] * b[0, 0] + a[3, 1] * b[1, 0] + a[3, 2] * b[2, 0] + a[3, 3] * b[3, 0]
    c31 = a[3, 0] * b[0, 1] + a[3, 1] * b[1, 1] + a[3, 2] * b[2, 1] + a[3, 3] * b[3, 1]
    c32 = a[3, 0] * b[0, 2] + a[3, 1] * b[1, 2] + a[3, 2] * b[2, 2] + a[3, 3] * b[3, 2]
    c33 = a[3, 0] * b[0, 3] + a[3, 1] * b[1, 3] + a[3, 2] * b[2, 3] + a[3, 3] * b[3, 3]

    return np.array([[c00, c01, c02, c03],
                     [c10, c11, c12, c13],
                     [c20, c21, c22, c23],
                     [c30, c31, c32, c33]])

def matmul3x1(a,b):
    c0 = a[0,0] * b[0] + a[0,1] * b[1] + a[0,2] * b[2]
    c1 = a[1,0] * b[0] + a[1,1] * b[1] + a[1,2] * b[2]
    c2 = a[2,0] * b[0] + a[2,1] * b[1] + a[2,2] * b[2]
    return np.array([c0,c1,c2])

def matmul4x1(a, b):
    c0 = a[0,0] * b[0] + a[0,1] * b[1] + a[0,2] * b[2] + a[0,3] * b[3]
    c1 = a[1,0] * b[0] + a[1,1] * b[1] + a[1,2] * b[2] + a[1,3] * b[3]
    c2 = a[2,0] * b[0] + a[2,1] * b[1] + a[2,2] * b[2] + a[2,3] * b[3]
    c3 = a[3,0] * b[0] + a[3,1] * b[1] + a[3,2] * b[2] + a[3,3] * b[3]
    return np.array([c0,c1,c2,c3])


def getTranslation(proj_mat, R, bbox2D, bbox3D):
    x_min, y_min, x_max, y_max = bbox2D
    w, h, l = bbox3D
    dx, dy, dz = w / 2., l / 2., h / 2.
    measure_max = -9999.
    trans_final = np.zeros(4)
    xmin_set_list = [[[-dx, -dy, -dz], [-dx, -dy, dz]], [[-dx, dy, -dz], [-dx, dy, dz]]]
    xmax_set_list = [[[dx, dy, -dz], [dx, dy, dz]], [[dx, -dy, dz], [dx, -dy, -dz]]]
    # ymin_set_list = [[[-dx, -dy, dz], [dx, -dy, dz]], [[-dx, dy, dz], [dx, dy, dz]]]
    # ymax_set_list = [[[-dx, dy, -dz], [dx, dy, -dz]], [[-dx, -dy, -dz], [dx, -dy, -dz]]]
    ymin_set_list = [[[-dx, -dy, dz], [dx, -dy, dz], [-dx, dy, dz], [dx, dy, dz]]]
    ymax_set_list = [[[-dx, dy, -dz], [dx, dy, -dz], [-dx, -dy, -dz], [dx, -dy, -dz]]]
    A0_set_list, A1_set_list, A2_set_list, A3_set_list = [], [], [], []
    B0_set_list, B1_set_list, B2_set_list, B3_set_list = [], [], [], []
    for xmin_set in xmin_set_list + xmax_set_list:
        A0_set, B0_set = [], []
        for d_xmin in xmin_set:
            A0 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_xmin), (3, 1))], axis=-1)
            A0 = np.concatenate([A0, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
            A0 = matmul4x4(proj_mat, A0)
            B0_set.append(A0)
            A0 = A0[0, :] - x_min * A0[2, :]
            A0_set.append(A0)
        A0_set_list.append(A0_set)
        B0_set_list.append(B0_set)
    for xmax_set in xmax_set_list + xmin_set_list:
        A2_set, B2_set = [], []
        for d_xmax in xmax_set:
            A2 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_xmax), (3, 1))], axis=-1)
            A2 = np.concatenate([A2, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
            A2 = matmul4x4(proj_mat, A2)
            B2_set.append(A2)
            A2 = A2[0, :] - x_max * A2[2, :]
            A2_set.append(A2)
        A2_set_list.append(A2_set)
        B2_set_list.append(B2_set)

    for ymin_set in ymin_set_list:
        A1_set, B1_set = [], []
        for d_ymin in ymin_set:
            A1 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_ymin), (3, 1))], axis=-1)
            A1 = np.concatenate([A1, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
            A1 = matmul4x4(proj_mat, A1)
            B1_set.append(A1)
            A1 = A1[1, :] - y_min * A1[2, :]
            A1_set.append(A1)
        A1_set_list.append(A1_set)
        B1_set_list.append(B1_set)
    for ymax_set in ymax_set_list:
        A3_set, B3_set = [], []
        for d_ymax in ymax_set:
            A3 = np.concatenate([np.identity(3), np.reshape(matmul3x1(R, d_ymax), (3, 1))], axis=-1)
            A3 = np.concatenate([A3, np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
            A3 = matmul4x4(proj_mat, A3)
            B3_set.append(A3)
            A3 = A3[1, :] - y_max * A3[2, :]
            A3_set.append(A3)
        A3_set_list.append(A3_set)
        B3_set_list.append(B3_set)

    for A0_set, B0_set, A2_set, B2_set in zip(A0_set_list, B0_set_list, A2_set_list, B2_set_list):
        for A1_set, B1_set, A3_set, B3_set in zip(A1_set_list, B1_set_list, A3_set_list, B3_set_list):
            for A0, B0 in zip(A0_set, B0_set):
                for A1, B1 in zip(A1_set, B1_set):
                    for A2, B2 in zip(A2_set, B2_set):
                        for A3, B3 in zip(A3_set, B3_set):
                            A = np.stack([A0, A1, A2, A3], axis=0)
                            # U, S, VH = scipy.linalg.svd(A)
                            U, S, VH = np.linalg.svd(A, full_matrices=True)
                            translation = VH[-1, :]
                            # translation = np.array([1.0, 1.0, 1.0, 1.0])
                            if translation[-1] * translation[-2] > 0:
                                translation = translation / translation[-1]
                                x_min_pred0 = matmul4x1(B0_set[0], translation)
                                x_min_pred0 = (x_min_pred0[:2] / x_min_pred0[2])[0]
                                x_min_pred1 = matmul4x1(B0_set[1], translation)
                                x_min_pred1 = (x_min_pred1[:2] / x_min_pred1[2])[0]
                                x_min_pred = np.min((x_min_pred0, x_min_pred1))

                                y_min_pred0 = matmul4x1(B1_set[0], translation)
                                y_min_pred0 = (y_min_pred0[:2] / y_min_pred0[2])[1]
                                y_min_pred1 = matmul4x1(B1_set[1], translation)
                                y_min_pred1 = (y_min_pred1[:2] / y_min_pred1[2])[1]
                                # y_min_pred = np.min((y_min_pred0, y_min_pred1))
                                y_min_pred2 = matmul4x1(B1_set[2], translation)
                                y_min_pred2 = (y_min_pred2[:2] / y_min_pred2[2])[1]
                                y_min_pred3 = matmul4x1(B1_set[3], translation)
                                y_min_pred3 = (y_min_pred3[:2] / y_min_pred3[2])[1]
                                y_min_pred = np.min((y_min_pred0, y_min_pred1, y_min_pred2, y_min_pred3))

                                x_max_pred0 = matmul4x1(B2_set[0], translation)
                                x_max_pred0 = (x_max_pred0[:2] / x_max_pred0[2])[0]
                                x_max_pred1 = matmul4x1(B2_set[1], translation)
                                x_max_pred1 = (x_max_pred1[:2] / x_max_pred1[2])[0]
                                x_max_pred = np.max((x_max_pred0, x_max_pred1))

                                y_max_pred0 = matmul4x1(B3_set[0], translation)
                                y_max_pred0 = (y_max_pred0[:2] / y_max_pred0[2])[1]
                                y_max_pred1 = matmul4x1(B3_set[1], translation)
                                y_max_pred1 = (y_max_pred1[:2] / y_max_pred1[2])[1]
                                # y_max_pred = np.max((y_max_pred0, y_max_pred1))
                                y_max_pred2 = matmul4x1(B3_set[2], translation)
                                y_max_pred2 = (y_max_pred2[:2] / y_max_pred2[2])[1]
                                y_max_pred3 = matmul4x1(B3_set[3], translation)
                                y_max_pred3 = (y_max_pred3[:2] / y_max_pred3[2])[1]
                                y_max_pred = np.max((y_max_pred0, y_max_pred1, y_max_pred2, y_max_pred3))

                                # if y_min<y_min_pred and y_max>y_max_pred:
                                if x_min_pred < x_max_pred and y_min_pred < y_max_pred:
                                    # if x_min_pred>=x_min and x_max_pred<=x_max:
                                    # if y_min_pred>=y_min and y_max_pred<=y_max:
                                    bbox2D_pred_area = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
                                    bbox2D_gt_area = (x_max - x_min) * (y_max - y_min)
                                    x_min_inter, x_max_inter = np.max((x_min_pred, x_min)), np.min((x_max_pred, x_max))
                                    y_min_inter, y_max_inter = np.max((y_min_pred, y_min)), np.min((y_max_pred, y_max))
                                    inter_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
                                    iou = inter_area / (bbox2D_pred_area + bbox2D_gt_area - inter_area)

                                    # x_min_outer, x_max_outer = np.min((x_min_pred, x_min)), np.max((x_max_pred, x_max))
                                    # y_min_outer, y_max_outer = np.min((y_min_pred, y_min)), np.max((y_max_pred, y_max))
                                    # d2 = np.square(x_max_outer-x_min_outer) + np.square(y_max_outer-y_min_outer)
                                    # cx_pred, cy_pred = (x_max_pred + x_min_pred)/2., (y_max_pred + y_min_pred)/2.
                                    # cx, cy = (x_max - x_min)/2., (y_max - y_min)/2.
                                    # c2 = np.square(cx_pred-cx) + np.square(cy_pred-cy)
                                    # rdiou = c2/d2
                                    # measure = iou - 0.5 * rdiou

                                    measure = iou
                                    if measure_max < measure:
                                        measure_max = measure
                                        trans_final = translation
                                        if measure > 0.9:
                                            return np.reshape(trans_final[:-1], (3, 1))

    return np.reshape(trans_final[:-1], (3, 1))

def getRay(P_inv, pixel):
    px, py = pixel
    pz = 1.0
    p_point = np.array([px,py,pz,1.])
    ray = matmul4x1(P_inv, p_point)
    ray = ray/ray[-1]
    ray = ray[:3]
    if ray[-1]<0:
        print('neg z', ray)
    # ray[1] -= 0.15
    return ray/np.linalg.norm(ray)

def getRayRotation(ray):
    ray = ray/np.linalg.norm(ray)
    # assume ray followed rotx and roty in order
    rx,ry,rz = ray
    cy = np.sqrt(ry*ry+rz*rz)
    cx = rz/np.sqrt(ry*ry+rz*rz)
    sx = -ry/np.sqrt(ry*ry+rz*rz)
    sy = rx
    R = np.array([[cy, 0., sy],
                 [sx*sy, cx, -sx*cy],
                 [-cx*sy, sx, cx*cy]])
    return R


def objRescaleTransform(objPoints, h, w, l, R):
    # pascal->kitti : 90rot for x axis R1, and -90rot for y axis R2
    R1 = np.array([[1, 0, 0, 0],
                   [0, 0,-1, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1]])
    R2 = np.array([[0, 0, -1, 0],
                   [0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])
    R = np.matmul(R, np.matmul(R2, R1))

    # R[:3,:3] = np.identity(3)

    dim = 64
    objPoints = objPoints.reshape(dim, dim, dim).astype('float')
    mask = objPoints > 0.5
    points = offset[mask]
    points = points[::2, :]  # sampling

    points = np.array(points).astype('float')
    points = points - np.min(points, axis=0)
    scale = np.max([h, w, l]) / np.max(points)
    points = points * scale
    points = points - np.max(points, axis=0) / 2.0

    points = np.transpose(np.concatenate([points, np.ones((len(points), 1))], axis=-1), [1, 0])
    rotPoints = np.transpose(np.matmul(R, points), [1, 0])[:, :3]
    # rotPoints = np.transpose(matmul4x4(R, points), [1, 0])[:, :3]
    return np.array(rotPoints)


def get3DbboxProjection(projmat, R, t, w, h, l):
    a = np.zeros((2, 2, 2, 2))
    bbox3D8Points = np.zeros((8,3))
    dx, dy, dz = -l / 2., -h / 2., -w / 2.  #car coordinate of kitti -> x,y,z : length, height, width (side view, 90 rotated view is the basic pose)
    for i in range(2):
        dx = -1. * dx
        for j in range(2):
            dy = -1. * dy
            for k in range(2):
                dz = -1. * dz
                x = matmul3x1(R, np.array([dx, dy, dz])) + np.reshape(t, (3,))
                x = np.array([x[0], x[1], x[2], 1.])
                x_proj = matmul4x1(projmat, x)
                x_proj = x_proj[:2] / x_proj[2]
                a[i, j, k, :] = x_proj
    return a

def get3Dbbox(R, t, w, h, l):
    a = np.zeros((2, 2, 2, 3))
    dx, dy, dz = -w / 2., -l / 2., -h / 2. # y and z are reversed
    for i in range(2):
        dx = -1. * dx
        for j in range(2):
            dy = -1. * dy
            for k in range(2):
                dz = -1. * dz
                x = matmul3x1(R, np.array([dx, dy, dz])) + np.reshape(t, (3,))
                a[i, j, k, :] = x
    return a


def draw2Dbbox(image, bbox2d, color=(0, 255, 0), thickness=2):
    p0 = (bbox2d[0], bbox2d[1])
    p1 = (bbox2d[2], bbox2d[3])
    cv2.rectangle(image, p0, p1, color=color, thickness=thickness)

def draw3Dbbox(image, proj_bbox3d, color=(255, 0, 255), thickness=2):
    proj_bbox3d = proj_bbox3d.astype('int32')
    # for cube
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[0, 1, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[1, 0, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 0, :]), tuple(proj_bbox3d[1, 1, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 0, 0, :]), tuple(proj_bbox3d[1, 1, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 0, :]), tuple(proj_bbox3d[0, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[0, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 1, 0, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 1, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 0, 0, :]), tuple(proj_bbox3d[1, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 1, :]), tuple(proj_bbox3d[0, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[1, 0, 1, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 1, :]), tuple(proj_bbox3d[1, 0, 1, :]), color=color, thickness=thickness)

    # X of forward and backward
    color = (255, 0, 0)
    thickness = 1
    cv2.line(image, tuple(proj_bbox3d[0, 0, 0, :]), tuple(proj_bbox3d[1, 0, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 0, 1, :]), tuple(proj_bbox3d[1, 0, 0, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 0, :]), tuple(proj_bbox3d[1, 1, 1, :]), color=color, thickness=thickness)
    cv2.line(image, tuple(proj_bbox3d[0, 1, 1, :]), tuple(proj_bbox3d[1, 1, 0, :]), color=color, thickness=thickness)

def getObjectInRealWorld(
        normalized_bbox2D_list, bbox3D_list, sin_list, cos_list, shape_3D_list,
        image_size_list,
        proj_mat_list, proj_mat_inv_list, localXYZ=None, localXYZ_log_var=None, isXYZ=False):

    # image_col, image_row = image_size
    objsPose, objsBbox3DSize, objsPoints = [], [], []
    objsBbox2D, objsBbox3DProj = [], []

    idx = 0
    for bbox2D, bbox3D, sin, cos, shape_3D, image_size, P2, P2_inv in zip(normalized_bbox2D_list, bbox3D_list, sin_list, cos_list, shape_3D_list, image_size_list, proj_mat_list, proj_mat_inv_list):
        # P2 = kitti_proj_mat
        # P2_inv = kitti_proj_mat_inv

        image_row, image_col = image_size
        b2x1,b2y1,b2x2,b2y2, _obj_prob = bbox2D
        # avoid too close obj
        # if b2x1 > 1e-1 and b2x2 < 1. - 1e-1 and b2y2 < 1. - 1e-1:
        if True:
            b2x1 = b2x1 * image_col
            b2y1 = b2y1 * image_row
            b2x2 = b2x2 * image_col
            b2y2 = b2y2 * image_row
            b3w,b3h,b3l = bbox3D
            sinA,sinE,sinI = sin
            cosA,cosE,cosI = cos

            E = np.arctan2(sinE,cosE) / np.pi * 180.0
            E = E * 0.01 * np.pi / 180.0
            sinE, cosE = np.sin(E), np.cos(E)
            # # sinI, cosI = 0., 1.

            # # elevation angle is wired. Manually enforcing
            # beta = -5.0 / 180.0 * np.pi
            # sinE_t = sinE*np.cos(beta) - cosE*np.sin(beta)
            # cosE_t = cosE*np.cos(beta) + sinE*np.sin(beta)
            # sinE = sinE_t
            # cosE = cosE_t

            # ======================== get rotation of obj
            # 1. RA*RE*RI
            r11, r12, r13 = -sinA * sinE * sinI + cosA * cosI, -sinA * cosE, sinA * sinE * cosI + sinI * cosA
            r21, r22, r23 = sinA * cosI + sinE * sinI * cosA, cosA * cosE, sinA * sinI - sinE * cosA * cosI
            r31, r32, r33 = -sinI * cosE, sinE, cosE * cosI

            # pascal->kitti : 90rot for x axis
            R = np.array([[r11, r12, r13],
                          [-r31, -r32, -r33],
                          [r21, r22, r23]])

            # apply ray orientation
            px, py = (b2x2 + b2x1) / 2., (b2y2 + b2y1) / 2.
            ray = getRay(P2_inv, (px, py))
            R_ray = getRayRotation(ray)
            R = matmul3x3(R_ray, R)

            # ======================== get translation of obj
            if localXYZ is None:
                X = getTranslation(P2, R, (b2x1, b2y1, b2x2, b2y2), (b3w, b3h, b3l))
            else:
                # X = np.reshape(localXYZ[idx], [3,1])
                if isXYZ:
                    X = np.reshape(localXYZ[idx], [3,1])
                else:
                    X = np.reshape(ray * localXYZ[idx], [3,1])
            # print(X)

            # ======================== append R,T of obj
            objPose = np.concatenate([np.concatenate([R, X], axis=-1), np.reshape([0, 0, 0, 1], (1, 4))], axis=0)

            # ======================== transform 3d shape of obj according to R,T
            objPoints = objRescaleTransform(shape_3D, b3h, b3w, b3l, objPose)

            # ====================== bbox3D projection on image
            proj_bbox3D = get3DbboxProjection(P2, R, X, b3w, b3h, b3l)

            # if translation != trivial solution (zero-vector) or too close to image plane, append
            if np.sqrt(np.sum(np.square(X))) > 1.:
                objsPose.append(objPose)
                objsPoints.append(objPoints)
                objsBbox2D.append([int(b2x1), int(b2y1), int(b2x2), int(b2y2)])
                objsBbox3DProj.append(proj_bbox3D)
                objsBbox3DSize.append([b3w, b3l, b3h])
            idx += 1

    objsPose = np.array(objsPose)
    objsBbox3DSize = np.array(objsBbox3DSize)
    objsPoints = np.array(objsPoints)
    objsBbox2D = np.array(objsBbox2D)
    objsBbox3DProj = np.array(objsBbox3DProj)

    return objsPose, objsBbox3DSize, objsPoints, objsBbox2D, objsBbox3DProj

def getXYZ(
        normalized_bbox2D_list, bbox3D_list, sin_list, cos_list, localXYZ_estimated_list,
        image_size,
        proj_mat=kitti_proj_mat, proj_mat_inv=kitti_proj_mat_inv):

    image_col, image_row = image_size
    XYZ, XYZ_estimated = [], []

    for bbox2D, bbox3D, sin, cos, localXYZ in zip(normalized_bbox2D_list, bbox3D_list, sin_list, cos_list, localXYZ_estimated_list):
        b2x1,b2y1,b2x2,b2y2, _obj_prob = bbox2D
        # avoid too close obj
        # if b2x1 > 1e-1 and b2x2 < 1. - 1e-1 and b2y2 < 1. - 1e-1:
        if True:
            b2x1 = b2x1 * image_col
            b2y1 = b2y1 * image_row
            b2x2 = b2x2 * image_col
            b2y2 = b2y2 * image_row
            b3w,b3h,b3l = bbox3D
            sinA,sinE,sinI = sin
            cosA,cosE,cosI = cos

            E = np.arctan2(sinE,cosE) / np.pi * 180.0
            E = E * 0.1 * np.pi / 180.0
            sinE, cosE = np.sin(E), np.cos(E)
            # sinE, cosE = 0.0, 1.0
            # # sinI, cosI = 0., 1.

            # # elevation angle is wired. Manually enforcing
            # beta = -5.0 / 180.0 * np.pi
            # sinE_t = sinE*np.cos(beta) - cosE*np.sin(beta)
            # cosE_t = cosE*np.cos(beta) + sinE*np.sin(beta)
            # sinE = sinE_t
            # cosE = cosE_t

            # ======================== get rotation of obj
            # 1. RA*RE*RI
            r11, r12, r13 = -sinA * sinE * sinI + cosA * cosI, -sinA * cosE, sinA * sinE * cosI + sinI * cosA
            r21, r22, r23 = sinA * cosI + sinE * sinI * cosA, cosA * cosE, sinA * sinI - sinE * cosA * cosI
            r31, r32, r33 = -sinI * cosE, sinE, cosE * cosI

            # pascal->kitti : 90rot for x axis
            R = np.array([[r11, r12, r13],
                          [-r31, -r32, -r33],
                          [r21, r22, r23]])

            # apply ray orientation
            px, py = (b2x2 + b2x1) / 2., (b2y2 + b2y1) / 2.
            ray = getRay(proj_mat_inv, (px, py))
            R_ray = getRayRotation(ray)
            R = matmul3x3(R_ray, R)

            # ======================== get translation of obj
            X = getTranslation(proj_mat, R, (b2x1, b2y1, b2x2, b2y2), (b3w, b3h, b3l))
            # print(X)
            X_estimated = np.reshape(ray * localXYZ, [3, 1])

            XYZ.append(X)
            XYZ_estimated.append(X_estimated)

    XYZ = np.array(XYZ)
    XYZ_estimated = np.array(XYZ_estimated)

    return XYZ, XYZ_estimated







