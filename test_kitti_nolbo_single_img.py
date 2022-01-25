import numpy as np
import time, sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

# from src.dataset_loader.KITTI_dataset import dataLoader
import src.net_core.darknet as Darknet
import src.module.nolbo_test as nolbo
import src.dataset_loader.KITTI_dataset as kitti
from src.visualizer.visualizer_ import *
import tensorflow as tf
import cv2
import scipy.io

import pangolin
import OpenGL.GL as gl

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')


def objMat2Elem(objMat, image_size, image_size_org, pred_num_per_grid, objCADs=None, projmat=None, projmat_inv=None):
    t = time.time()
    objs = objMat['record']['objects'][0][0][0]
    objsPose, objsBbox3DSize, objsPoints = [], [], []
    objsBbox2D = []
    objsBbox3DProj = []
    print('objMat : ', time.time() - t)

    image_row, image_col = image_size
    image_row_org, image_col_org = image_size_org
    grid_row, grid_col = image_row // 32, image_col // 32
    objnessImage = np.zeros([grid_row, grid_col, pred_num_per_grid, 1])
    bbox2DImage = np.zeros([grid_row, grid_col, pred_num_per_grid, 4])
    bbox3DImage = np.zeros([grid_row, grid_col, pred_num_per_grid, 3])
    localXYZImage = np.zeros([grid_row, grid_col, pred_num_per_grid, 3])
    eulerRadImage = np.zeros([grid_row, grid_col, pred_num_per_grid, 3])

    for obj in objs:
        if obj[0][0] == 'Car':
            print('#=========================start===========================#')
            start_time = time.time()
            truncated = obj[1][0][0]
            occluded = obj[2][0][0]
            b2x1, b2y1, b2x2, b2y2 = obj[4][0][0], obj[5][0][0], obj[6][0][0], obj[7][0][0]
            b3h, b3w, b3l = obj[8][0][0], obj[9][0][0], obj[10][0][0]
            loc_xyz = np.array(obj[11][0])
            azimuth, elevation = obj[14][0][0], obj[15][0][0]
            azimuth, elevation, inPlaneRot = -float(azimuth) / 180.0 * np.pi, -float(elevation) / 180.0 * np.pi, 0.0
            objBbox3DSize = np.array([b3h, b3l, b3w])
            objsBbox3DSize.append(objBbox3DSize)
            objsBbox2D.append([int(b2x1), int(b2y1), int(b2x2), int(b2y2)])
            rowMin, colMin, rowMax, colMax = b2y1, b2x1, b2y2, b2x2
            print((rowMin+rowMax)/2., (colMin+colMax)/2.)
            rowCenterOnGrid = (rowMax+rowMin)/2.0 * grid_row / image_row_org
            colCenterOnGrid = (colMax+colMin)/2.0 * grid_col / image_col_org
            rowIndexOnGrid = int(rowCenterOnGrid)
            colIndexOnGrid = int(colCenterOnGrid)
            dx, dy = colCenterOnGrid - colIndexOnGrid, rowCenterOnGrid - rowIndexOnGrid
            bboxHeight, bboxWidth = (rowMax - rowMin)/ image_row_org, (colMax- colMin)/ image_col_org
            for predIndex in range(pred_num_per_grid):
                if objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex, 0] != 1:
                    objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex, 0] = 1
                    bbox2DImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = bboxHeight, bboxWidth, dx, dy
                    bbox3DImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = b3w, b3h, b3l
                    localXYZImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = loc_xyz[0], loc_xyz[1]-b3h/2., loc_xyz[2]
                    eulerRadImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = azimuth, elevation, inPlaneRot
                    break
    objnessImages = []
    bbox2DImages = []
    bbox3DImages = []
    localXYZImages = []
    eulerRadImages = []
    objnessImages.append(objnessImage)
    bbox2DImages.append(bbox2DImage)
    bbox3DImages.append(bbox3DImage)
    localXYZImages.append(localXYZImage)
    eulerRadImages.append(eulerRadImage)
    objnessImages = np.array(objnessImages)
    bbox2DImages = np.array(bbox2DImages)
    bbox3DImages = np.array(bbox3DImages)
    localXYZImages = np.array(localXYZImages)
    eulerRadImages = np.array(eulerRadImages)
    #         loc_xyz = np.array(obj[11][0])
    #         ry = obj[12][0][0]
    #         cad_index = int(obj[13][0][0])  # [1,2,3,4,5,6,7] -> [1,2,3,4,5,6,9]
    #         if cad_index == 7:
    #             cad_index = 9
    #         objCAD = objCADs[cad_index]
    #         azimuth, elevation = obj[14][0][0], obj[15][0][0]
    #         azimuth, elevation, inPlaneRot = -(float(azimuth)) / 180.0 * np.pi, -float(
    #             elevation - 5) / 180.0 * np.pi, 0.0
    #         #             azimuth,elevation,inPlaneRot = -(float(azimuth))/180.0*np.pi, 0.0, 0.0
    #         #             azimuth,elevation,inPlaneRot = (-float(ry + np.pi/2.)), 0., 0.
    #
    #         t = time.time()
    #         sinA, cosA = np.sin(azimuth), np.cos(azimuth)
    #         sinE, cosE = np.sin(elevation), np.cos(elevation)
    #         sinI, cosI = np.sin(inPlaneRot), np.cos(inPlaneRot)
    #
    #         # RA*RE*RI -> this one better
    #         r11, r12, r13 = -sinA * sinE * sinI + cosA * cosI, -sinA * cosE, sinA * sinE * cosI + sinI * cosA
    #         r21, r22, r23 = sinA * cosI + sinE * sinI * cosA, cosA * cosE, sinA * sinI - sinE * cosA * cosI
    #         r31, r32, r33 = -sinI * cosE, sinE, cosE * cosI
    #
    #         # RI*RE*RA
    #         #             r11, r12, r13 = cosA*cosI+sinA*sinE*sinI, sinA*sinE*sinI-sinA*cosI, cosE*sinI
    #         #             r21, r22, r23 = sinA*cosE, cosA*cosE, -sinE
    #         #             r31, r32, r33 = sinA*sinE*cosI-cosA*sinI, cosA*sinE*cosI+sinA*sinI, cosE*cosI
    #
    #         R = np.array([[r11, r12, r13],
    #                       [r21, r22, r23],
    #                       [r31, r32, r33]])
    #         # azimuth,elevation,inplanerot are of camera's orientation, so convert to obj's
    #         #             R = np.linalg.inv(R)
    #         print('R', time.time() - t)
    #
    #         t = time.time()
    #         # pascal->kitti : 90rot for x axis
    #         #             Rx90 = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    #         #             R = np.matmul(Rx90, R)
    #         # too slow, so swith to hard code
    #         R = np.array([[r11, r12, r13],
    #                       [-r31, -r32, -r33],
    #                       [r21, r22, r23]])
    #         print('Rmatmul', time.time() - t)
    #
    #         t = time.time()
    #         # get ray orientation
    #         px, py = (b2x2 + b2x1) / 2., (b2y2 + b2y1) / 2.
    #         ray = getRay(projmat_inv, (px, py))
    #         R_ray = getRayRotation(ray)
    #         print('get R ray', time.time() - t)
    #         t = time.time()
    #         # apply ray direction
    #         #             R = np.matmul(R_ray, R)
    #         #             R = np.dot(R_ray, R)
    #         R = matmul3x3(R_ray, R)
    #         print('R ray matmul', time.time() - t)
    #
    #         X = np.reshape(np.array([loc_xyz[0], loc_xyz[1], loc_xyz[2]]), (3, 1))
    #         # #             locations are already in camera(or image) coordinate
    #         #             print((b2x2+b2x1)/2., (b2y2+b2y1)/2.)
    #         #             xt = np.reshape(np.array([loc_xyz[0], loc_xyz[1], loc_xyz[2], 1]), (4,1))
    #         #             xt = np.matmul(projmat, xt)
    #         #             xt = xt/xt[2]
    #         #             print(xt)
    #         #             print(X)
    #         t = time.time()
    #         X = getTranslation(projmat, R, (b2x1, b2y1, b2x2, b2y2), (b3w, b3h, b3l))
    #         #             print(X)
    #         print('get translation', time.time() - t)
    #
    #         t = time.time()
    #         proj_bbox3D = get3DbboxProjection(projmat, R, X, b3h, b3w, b3l)
    #         objsBbox3DProj.append(proj_bbox3D)
    #         print('3d bbox projection', time.time() - t)
    #
    #         objPose = np.concatenate(
    #             [np.concatenate([R, X], axis=-1), np.reshape([0, 0, 0, 1], (1, 4))], axis=0)
    #
    #         objsPose.append(objPose)
    #
    #         t = time.time()
    #         objPoints = objRescaleTransform(objCAD, b3h, b3w, b3l, objPose)
    #         objsPoints.append(objPoints)
    #         print('objRescaleTransform', time.time() - t)
    #         print('total', time.time() - start_time)
    #
    # return np.array(objsPose), np.array(objsBbox3DSize), np.array(objsPoints), np.array(objsBbox2D), np.array(
    #     objsBbox3DProj)
    return objnessImages, bbox2DImages, bbox3DImages, localXYZImages, np.sin(eulerRadImages), np.cos(eulerRadImages)


predictor_num = 2
latent_dim = 16
inst_num = 10
config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num': predictor_num,
        'bbox2D_dim':4, 'bbox3D_dim':3, 'localXYZ_dim':1, 'orientation_dim':3,
        'inst_dim':10, 'z_inst_dim':16,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : predictor_num*(1+4+3+(2*3+3)+2*1+2*latent_dim),  # objness + hwxy + whl + (2*sincosAEI+radAEI) + 2*latent
        'filter_num_list':[1024,1024,1024],
        'filter_size_list':[3,3,3],
        'activation':'elu',
    },
    'decoder':{
        'name':'decoder',
        'input_dim' : latent_dim,
        'output_shape':[64,64,64,1],
        'filter_num_list':[512,256,128,64,1],
        'filter_size_list':[4,4,4,4,4],
        'strides_list':[1,2,2,2,2],
        'activation':'elu',
        'final_activation':'sigmoid'
    },
    'prior' : {
        'name' : 'priornet',
        'input_dim' : inst_num,  # class num (one-hot vector)
        'unit_num_list' : [32, 16],
        'core_activation' : 'elu',
        'const_log_var' : 0.0,
    }
}

model = nolbo.nolboXYZEval(nolbo_structure=config,
                           anchor_boxes=kitti.anchor_boxes_k2_edist,
                           anchor_distance=kitti.anchor_dist_k2_edist,
                           backbone_style=Darknet.Darknet19)
model.loadModel(load_path='./weights_dist_exp/k2_edist/')
image_size_network = (1216, 384)
# image_size_network = (960, 352)
image_size_visualize = (1216, 384)

# prepare pangolin visualizer
pangolin.CreateWindowAndBind('Main', image_size_visualize[0], image_size_visualize[1] * 2)
gl.glEnable(gl.GL_DEPTH_TEST)

# Define Projection and initial ModelView matrix
proj = pangolin.ProjectionMatrix(image_size_visualize[0], image_size_visualize[1], 718.856, 718.856, 620, 188, 0.2, 200)
# scam = pangolin.OpenGlRenderState(
#     proj, pangolin.ModelViewLookAt(0, -2, -10, 0, 0, 10, pangolin.AxisDirection.AxisNegY))
scam = pangolin.OpenGlRenderState(
    proj, pangolin.ModelViewLookAt(0, -140, 10, 0, 0, 35.1, pangolin.AxisDirection.AxisNegY))
# scam = pangolin.OpenGlRenderState(
#     proj, pangolin.ModelViewLookAt(0, -70, 15, 0, 0, 20.1, pangolin.AxisDirection.AxisNegY))

# Create Interactive View in window
dcam = pangolin.Display('cam1')
dcam.SetBounds(0.0, 1.0, 0.0, 1.0, - 1.0 * image_size_visualize[0] / image_size_visualize[1])  # bound yx start-end
dcam.SetAspect(1.0 * image_size_visualize[0] / image_size_visualize[1])
dcam.SetHandler(pangolin.Handler3D(scam))

dImg = pangolin.Display('img1')
dImg.SetBounds(0.0, 1.0, 0.0, 1.0, 1.0 * image_size_visualize[0] / image_size_visualize[1])
dImg.SetAspect(1.0 * image_size_visualize[0] / image_size_visualize[1])
# dImg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

view = pangolin.Display('multi')
view.SetBounds(0.0, 1.0, 0.0, 1.0)
view.SetLayout(pangolin.LayoutEqual)
view.AddDisplay(dImg)
view.AddDisplay(dcam)
image_texture = pangolin.GlTexture(image_size_visualize[0], image_size_visualize[1], gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

panel = pangolin.CreatePanel('ui')
panel.SetBounds(0.0, 65./image_size_visualize[1], 0.0, 125./image_size_visualize[0])
check_world_shape = pangolin.VarBool('ui.world_shape', value=True, toggle=True)
check_world_bbox3D = pangolin.VarBool('ui.world_bbox3D', value=True, toggle=True)
check_image_bbox2D = pangolin.VarBool('ui.image_bbox2D', value=True, toggle=True)
check_image_bbox3D = pangolin.VarBool('ui.image_bbox3D', value=False, toggle=True)

# file_name = '{:06d}'.format(278)
# file_name = '{:06d}'.format(260)
file_name = '{:06d}'.format(38)
# file_name = '{:06d}'.format(26)
image_path = '/media/yonsei/4TB_HDD/downloads/KITTI/data_object_image_2/training/image_2/'
image = cv2.imread(image_path+file_name+'.png', cv2.IMREAD_COLOR)
image_size = image.shape[0], image.shape[1]
image_size_list = np.array([image_size])
label_path = '/media/yonsei/4TB_HDD/downloads/KITTI/3DVP_Annotations/Annotations/'
objMat = scipy.io.loadmat(label_path+file_name+'.mat')
KITTI_calib_path='/home/yonsei/dataset/KITTI/data_object_calib/training/calib/'
with open(os.path.join(KITTI_calib_path, file_name + '.txt')) as fp:
    calib = fp.readlines()
P2 = np.array(calib[2].split(' ')[1:])
projection_mat = np.identity(4)
for i in range(3):
    for j in range(4):
        projection_mat[i, j] = float(P2[4 * i + j])
projection_mat_inv = np.linalg.inv(projection_mat)
P2_list = np.array([projection_mat])
P2_inv_list = np.array([projection_mat_inv])

objnessImages, bbox2DImages, bbox3DImages, localXYZImages, sin_gt, cos_gt = objMat2Elem(
    objMat=objMat, image_size=(image_size_network[1], image_size_network[0]), image_size_org=image_size, pred_num_per_grid=predictor_num)
print('load done')
while (not pangolin.ShouldQuit()):
    start_time = time.time()

    image_input = cv2.resize(image, image_size_network) / 255.0
    inputImages = []
    inputImages.append(image_input)
    inputImages = np.array(inputImages)
    image_visualize = cv2.resize(image, (image_size[1], image_size[0]))


    # def getEval(self, inputImages, objnessImages, bbox2DImages, bbox3D_gt, localXYZImages, sin_gt, cos_gt,
    #             imageSizeList, P2List, P2InvList,
    #             image_reduced=32):

    objsPose, objsBbox3DSize, objsPoints, objsBbox2D, objsBbox3DProj, \
    objsPose_cal, objsBbox3DSize_cal, objsPoints_cal, objsBbox2D_cal, objsBbox3DProj_cal,\
    objsPose_gt, objsBbox3DSize_gt, objsPoints_gt, objsBbox2D_gt, objsBbox3DProj_gt = model.getEval(
        inputImages=inputImages,objnessImages=objnessImages, bbox2DImages=bbox2DImages, bbox3D_gt=bbox3DImages, localXYZImages=localXYZImages,
        sin_gt=sin_gt, cos_gt=cos_gt,
        imageSizeList=image_size_list, P2List=P2_list, P2InvList=P2_inv_list,
        isWorld=True
    )

    # # objsPose, objsBbox3DSize, objsPoints, objsBbox2D, objsBbox3DProj = visualizer.getObjectInRealWorld(
    # #     normalized_bbox2D_list=bbox2D_selected,
    # #     bbox3D_list=bbox3D_selected,
    # #     sin_list=sin_selected, cos_list=cos_selected,
    # #     shape_3D_list=outputs_3D_shape,
    # #     image_size=(inputImgCol, inputImgRow),
    # #     localXYZ=localXYZ_selected, localXYZ_log_var=localXYZ_log_var_selected
    # # )

    # objsPose, objsBbox3DSize, objsPoints, objsBbox2D, objsBbox3DProj= [], [], [], [], []

    if len(objsBbox2D_gt)>0:
        if check_image_bbox2D.Get():
            for objBbox2D in objsBbox2D_gt:
                draw2Dbbox(image_visualize, objBbox2D)
        if check_image_bbox3D.Get():
            for objBbox3DProj in objsBbox3DProj_gt:
                draw3Dbbox(image_visualize, objBbox3DProj)

    # if len(objsBbox2D)>0:
    #     if check_image_bbox2D.Get():
    #         for objBbox2D in objsBbox2D:
    #             draw2Dbbox(image_visualize, objBbox2D)
    #     if check_image_bbox3D.Get():
    #         for objBbox3DProj in objsBbox3DProj:
    #             draw3Dbbox(image_visualize, objBbox3DProj)

    # for pangolin, don't know why I should do below
    image_visualize = cv2.resize(image_visualize, image_size_visualize)
    image_visualize = cv2.cvtColor(image_visualize, cv2.COLOR_BGR2RGB)
    image_visualize = cv2.flip(image_visualize, 0)

    bboxPoses = objsPose
    bboxSizes = objsBbox3DSize
    points = objsPoints

    bboxPoses_cal = objsPose_cal
    bboxSizes_cal = objsBbox3DSize_cal
    points_cal = objsPoints_cal

    bboxPoses_gt = objsPose_gt
    bboxSizes_gt = objsBbox3DSize_gt
    points_gt = objsPoints_gt

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)

    dcam.Activate(scam)
    # Draw lines
    gl.glLineWidth(10)
    gl.glColor3f(1.0, 0.0, 0.0)
    pangolin.DrawLine(np.array([[0, 0, 0], [1, 0, 0]]))
    # Draw lines
    gl.glLineWidth(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawLine(np.array([[0, 0, 0], [0, 1, 0]]))
    # Draw lines
    gl.glLineWidth(10)
    gl.glColor3f(0.0, 0.0, 1.0)
    pangolin.DrawLine(np.array([[0, 0, 0], [0, 0, 1]]))

    ##     Render OpenGL Cube
    #     pangolin.glDrawColouredCube(0.1)

    #     # Draw point cloud
    #     gl.glPointSize(3)
    #     gl.glColor3f(1.0, 0.0, 0.0)
    #     pangolin.DrawPoints(points)

    # if len(objsPoints) > 0:
    #     if check_world_shape.Get():
    #         # Draw Objects
    #         for objPoints in objsPoints:
    #             gl.glPointSize(2)
    #             gl.glColor3f(1.0, 0.0, 0.0)
    #             pangolin.DrawPoints(objPoints)
    #     if check_world_bbox3D.Get():
    #         # Draw bbox
    #         gl.glLineWidth(1)
    #         gl.glColor3f(1.0, 0.0, 0.0)
    #         pangolin.DrawBoxes(bboxPoses, bboxSizes)
    #
    # if len(objsPoints_cal) > 0:
    #     if check_world_shape.Get():
    #         # Draw Objects
    #         for objPoints in objsPoints_cal:
    #             gl.glPointSize(2)
    #             gl.glColor3f(1.0, 0.0, 0.0)
    #             pangolin.DrawPoints(objPoints)
    #     if check_world_bbox3D.Get():
    #         # Draw bbox
    #         gl.glLineWidth(1)
    #         gl.glColor3f(1.0, 0.0, 0.0)
    #         pangolin.DrawBoxes(bboxPoses_cal, bboxSizes_cal)

    if len(objsPoints_gt) > 0:
        if check_world_shape.Get():
            # Draw Objects
            for objPoints in objsPoints_gt:
                gl.glPointSize(2)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(objPoints)
        if check_world_bbox3D.Get():
            # Draw bbox
            gl.glLineWidth(1)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawBoxes(bboxPoses_gt, bboxSizes_gt)

    end_time = time.time()
    run_time = end_time - start_time
    FPS = 1./ run_time

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (0, 12)
    # fontScale = 0.5
    # fontColor = (0, 0, 0)
    # lineType = 2
    # text = 'FPS : {:.2f}'.format(FPS)
    # (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
    # text_offset_x = 0
    # text_offset_y = 15
    # box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    #
    # image_visualize = cv2.flip(image_visualize, 0)
    # rectangle_bgr = (255, 255, 255)
    # cv2.rectangle(image_visualize, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    # cv2.putText(image_visualize, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    # image_visualize = cv2.flip(image_visualize, 0)

    image_texture.Upload(image_visualize, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    dImg.Activate()
    gl.glColor3f(1.0, 1.0, 1.0)
    image_texture.RenderToViewport()

    pangolin.FinishFrame()

