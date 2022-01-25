import numpy as np
import time, sys, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sys.path.append('/home/yonsei/Downloads/pangolin/pangolin.cpython-35m-x86_64-linux-gnu.so')
# sys.path.append('/home/yonsei/Downloads/pangolin/pangolin.cpython-36m-x86_64-linux-gnu.so')
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

import pangolin
import OpenGL.GL as gl

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

predictor_num = 12
config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num': predictor_num,
        'bbox2DXY_dim':2,
        'bbox3D_dim':3, 'orientation_dim':1,
        'localXYZ_dim':1,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : predictor_num*(1+2+3+1+2),  # objness + b2Dxy + lhw + z + sincos
        'filter_num_list':[1024,1024,1024],
        'filter_size_list':[3,3,3],
        'activation':'elu',
    }
}
anchor_path = './data/kitti_split1_org'
weight_path = './weights/kitti_split1_org/'

image_size_network = (1216,448)
image_size_visualize = (1200, 300)
model = nolbo.nolbo_test(nolbo_structure=config, backbone_style=Darknet.Darknet19)
model.loadModel(load_path=weight_path)

anchor_z_global = np.load(os.path.join(anchor_path, 'global_anchor_z_s.npy'))
anchor_bbox3D = np.load(os.path.join(anchor_path, 'anchor_bbox3D.npy'))

datasetPath = '/media/yonsei/4TB_HDD/dataset/kitti/'
data_index = 2
# data_index = 0
calibPath = os.path.join(datasetPath, 'KITTI_odometry_RGB/{:02d}/calib.txt'.format(data_index))
with open(calibPath) as fp:
    calib = fp.readlines()
P2 = np.array(calib[2].split(' ')[1:]).astype('float64')
projection_mat = np.identity(4)
projection_mat_inv = np.identity(4)
for i in range(3):
    for j in range(4):
        projection_mat[i, j] = P2[4 * i + j]
P2_pinv = np.linalg.pinv(projection_mat)
for i in range(4):
    for j in range(3):
        projection_mat_inv[i, j] = P2_pinv[i, j]
# projection_mat_inv = np.linalg.inv(projection_mat)
P2_list = np.array([projection_mat])
P2_inv_list = np.array([projection_mat_inv])
dataImgDirPath = os.path.join(datasetPath, 'KITTI_odometry_RGB/{:02d}/image_3'.format(data_index))
fileList = os.listdir(dataImgDirPath)
fileList.sort()
imgList = []
for fileName in fileList:
    if fileName.endswith(".png") or fileName.endswith(".jpg"):
        imgList.append(fileName)


# prepare pangolin visualizer
pangolin.CreateWindowAndBind('Main', image_size_visualize[0], image_size_visualize[1] * 2)
gl.glEnable(gl.GL_DEPTH_TEST)

# Define Projection and initial ModelView matrix
proj = pangolin.ProjectionMatrix(image_size_visualize[0], image_size_visualize[1], 718.856, 718.856, 620, 188, 0.2, 200)
# scam = pangolin.OpenGlRenderState(
#     proj, pangolin.ModelViewLookAt(0, -2, -10, 0, 0, 10, pangolin.AxisDirection.AxisNegY))
scam = pangolin.OpenGlRenderState(
    proj, pangolin.ModelViewLookAt(0, -100, -50, 0, 0, 10.1, pangolin.AxisDirection.AxisNegY))

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
check_image_bbox3D = pangolin.VarBool('ui.image_bbox3D', value=True, toggle=True)

# image = cv2.imread('data/000021.png', cv2.IMREAD_COLOR)
#
# image_input1 = cv2.resize(image, image_size_network)
# image_visualize1 = cv2.resize(image, image_size_visualize)

image_index = 0
while (not pangolin.ShouldQuit()) and (image_index < len(imgList)):
    start_time = time.time()
    image_path = os.path.join(dataImgDirPath, imgList[image_index])
    image_index += 1
    image = cv2.imread(image_path , cv2.IMREAD_COLOR)
    image_size = image.shape[1], image.shape[0]
    image_input = cv2.resize(image, image_size_network)
    image_visualize = cv2.resize(image, image_size)

    objsPose, objsBbox3DSize, bbox3D8Points, objsBbox2D, rys, alphas, objnesses = \
    model.getPred(
        input_images=image_input,
        P2_gt=projection_mat, P2_inv_gt=projection_mat_inv,
        image_size_org=image_size, obj_thresh=0.95,
        anchor_z=anchor_z_global, anchor_bbox3D=anchor_bbox3D
    )
    # objsPose, objsBbox3DSize, objsPoints, objsBbox2D, objsBbox3DProj= [], [], [], [], []

    if len(objsBbox2D)>0:
        if check_image_bbox2D.Get():
            for objBbox2D in objsBbox2D:
                draw2Dbbox(image_visualize, objBbox2D)
        # if check_image_bbox3D.Get():
        #     for objBbox3DProj in objsBbox3DProj:
        #         draw3Dbbox(image_visualize, objBbox3DProj)

    # for pangolin, don't know why I should do below
    image_visualize = cv2.resize(image_visualize, image_size_visualize)
    image_visualize = cv2.cvtColor(image_visualize, cv2.COLOR_BGR2RGB)
    image_visualize = cv2.flip(image_visualize, 0)

    bboxPoses = objsPose
    bboxSizes = objsBbox3DSize
    # points = objsPoints

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

    if len(bboxPoses) > 0:
        # if check_world_shape.Get():
        #     # Draw Objects
        #     for objPoints in objsPoints:
        #         gl.glPointSize(2)
        #         gl.glColor3f(1.0, 0.0, 0.0)
        #         pangolin.DrawPoints(objPoints)
        if check_world_bbox3D.Get():
            # Draw bbox
            gl.glLineWidth(1)
            gl.glColor3f(1.0, 0.0, 1.0)
            pangolin.DrawBoxes(bboxPoses, bboxSizes)

    end_time = time.time()
    run_time = end_time - start_time
    FPS = 1./ run_time

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0, 12)
    fontScale = 0.5
    fontColor = (0, 0, 0)
    lineType = 2
    text = 'FPS : {:.2f}'.format(FPS)
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
    text_offset_x = 0
    text_offset_y = 15
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))

    image_visualize = cv2.flip(image_visualize, 0)
    rectangle_bgr = (255, 255, 255)
    cv2.rectangle(image_visualize, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image_visualize, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    image_visualize = cv2.flip(image_visualize, 0)

    image_texture.Upload(image_visualize, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    dImg.Activate()
    gl.glColor3f(1.0, 1.0, 1.0)
    image_texture.RenderToViewport()

    pangolin.FinishFrame()

