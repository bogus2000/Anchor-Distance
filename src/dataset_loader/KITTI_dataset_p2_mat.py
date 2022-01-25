import scipy.io
import os
import numpy as np
import cv2
import math
from sklearn.cluster import KMeans
from src.module.function import kmeans_dist
from src.dataset_loader.datasetUtils import *
from src.visualizer.visualizer_ import *
# from utils import imageAugmentation
# from utils import voxelBlur

'''
    In this code, I assume that object center is 3D bbox center, not the bottom as in KITTI dataset paper.
    viewpoint is 1 dimension (roll angle only)
'''

category_to_index = {
    'Car':0,
    'Pedestrian':1,
    'Cyclist':2,
    'Van':3,
    'Truck':4,
    'Person_sitting':5,
    'Tram':6,
    'Misc':7,
    'DontCare':8,
}
category_num = len(category_to_index) - 1
car_instance_num = 10

class dataLoader(object):
    def __init__(self,
                 imageSize=(1280, 384),
                 gridSize=(40, 12),
                 predNumPerGrid=5,
                 KITTIAnnotPath='/home/yonsei/dataset/KITTI/3DVP_Annotations/Annotations/',
                 KITTIImagePath='/home/yonsei/dataset/KITTI/data_object_image_2/training/image_2/',
                 KITTI_calib_path='/home/yonsei/dataset/KITTI/data_object_calib/training/calib/',
                 KITTI3DShapePath='/home/yonsei/dataset/PASCAL3D+_release1.1/CAD/car/',
                 KITTI_anchor_path='',
                 is_train=True,
                 ):
        self.dataStart = 0
        self.dataLength = 0
        self.epoch = 0

        self._imageSize = imageSize
        self._gridSize = gridSize
        self._predNumPerGrid = predNumPerGrid
        self._KITTIAnnotPath = KITTIAnnotPath
        self._KITTIImagePath = KITTIImagePath
        self._KITTI_calib_path = KITTI_calib_path
        self._KITTI3DShapePath = KITTI3DShapePath
        self._KITTI_anchor_path = KITTI_anchor_path
        self._isTrain = is_train
        self._filename_list = np.sort(os.listdir(self._KITTIImagePath))
        for i, filename in enumerate(self._filename_list):
            self._filename_list[i] = filename.split('.')[0]
        self.dataLength = len(self._filename_list)
        self._KITTI3DShapes = []

        print('set kitti dataset...')
        self._load3DShapes()
        try:
            self.loadPrior()
        except:
            self.get3DboxAverage(save_path=self._KITTI_anchor_path)
            self.getKMeansDist(k=12, max_iter=1000, save_path=self._KITTI_anchor_path)
            self.getImageAverage(save_path=self._KITTI_anchor_path)
            self.loadPrior()

        if self._isTrain:
            self._dataShuffle()

    def _dataShuffle(self):
        # print('data path shuffle...')
        self.dataStart = 0
        np.random.shuffle(self._filename_list)
        # print('done! : ' + str(self.dataLength))

    def _load3DShapes(self):
        print('load 3d shapes for KITTI...')
        self._KITTI3DShapes = []
        CADModelList = os.listdir(self._KITTI3DShapePath)
        CADModelList.sort()
        # print CADModelList
        for CADModel in CADModelList:
            if CADModel.split(".")[-1] == 'npy':
                shape = np.load(os.path.join(self._KITTI3DShapePath, CADModel)).reshape(64, 64, 64, 1)
                self._KITTI3DShapes.append(shape)
        self._KITTI3DShapes = np.array(self._KITTI3DShapes)
        self._KITTI3DShapes = np.where(self._KITTI3DShapes>0, 1.0, 0.0)
        print('shape num:', len(self._KITTI3DShapes))
        print('done!')

    def _get3DbboxProjection(self, projmat, R, t, w, h, l):
        a = np.zeros((2, 2, 2, 2))
        bbox3D8Points = []
        dx, dy, dz = -l / 2., -h / 2., -w / 2.  # car coordinate of kitti -> x,y,z : length, height, width (side view, 90 rotated view is the basic pose)
        for i in range(2):
            dy = -1. * dy
            for j in range(2):
                dx = -1. * dx
                for k in range(2):
                    dz = -1. * dz
                    x = matmul3x1(R, np.array([dx, dy, dz])) + np.reshape(t, (3,))
                    bbox3D8Points.append(x)
                    x = np.array([x[0], x[1], x[2], 1.])
                    x_proj = matmul4x1(projmat, x)
                    x_proj = x_proj[:2] / x_proj[2]
                    a[i, j, k, :] = x_proj
        # print(np.array(bbox3D8Points).shape)
        return a, np.array(bbox3D8Points)

    def _getRay(self, P_inv, pixel):
        px, py = pixel
        pz = 1.0
        p_point = np.array([px, py, pz, 1.])
        ray = matmul4x1(P_inv, p_point)
        ray = ray / ray[-1]
        ray = ray[:3]
        if ray[-1] < 0:
            print('neg z', ray)
        # ray[1] -= 0.15
        return ray / np.sqrt(np.sum(np.square(ray)))

    def _convertRot2Alpha(self, ry3d, z3d, x3d):

        alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
        # alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi

        while alpha > math.pi: alpha -= math.pi * 2
        while alpha < (-math.pi): alpha += math.pi * 2

        return alpha

    def _objMat2Elem(self, annot_path, P2, P2_inv, isFlip=False):
        objMat = scipy.io.loadmat(annot_path)
        image_col, image_row, _ = objMat['record']['imgsize'][0][0][0]
        objs = objMat['record']['objects'][0][0][0]
        objsInfo = []
        depth = []
        for obj in objs:
            # obj = obj.split(' ')
            category_name = obj[0][0]
            if category_name != 'DontCare':
                truncated = float(obj[1][0][0])
                occluded = float(obj[2][0][0])
                alpha = float(obj[3][0][0])
                x1, y1, x2, y2 = float(obj[4][0][0]), float(obj[5][0][0]), float(obj[6][0][0]), float(obj[7][0][0])
                h, w, l = float(obj[8][0][0]), float(obj[9][0][0]), float(obj[10][0][0])
                local_xyz = np.array(obj[11][0])
                ry = float(obj[12][0][0])
                alpha = self._convertRot2Alpha(ry3d=ry, x3d=local_xyz[0], z3d=local_xyz[2])
                cad_index = -1
                if category_name == 'Car':
                    cad_index = int(obj[13][0][0])
                    if cad_index==7:
                        cad_index = 9

                local_xyz[1] = local_xyz[1] - h / 2.  # 3d bbox center is object center, not bottom

                center_proj = np.array([local_xyz[0], local_xyz[1], local_xyz[2], 1.]).astype('float128')
                center_proj = matmul4x1(P2, center_proj)
                center_proj = center_proj[:2] / center_proj[2]

                if isFlip:
                    # need to change the orientation
                    if ry > 0:
                        ry = np.pi - ry
                    else:
                        ry = -np.pi - ry
                    # flip the x of pixel and reproject the pixel to 3D world
                    center_proj[0] = image_col - (center_proj[0] - 1)
                    # ray = self._getRay(P2_inv, center_proj)
                    # local_xyz = ray / ray[-1] * local_xyz[-1] # only depth remains same
                    local_xyz = matmul4x1(P2_inv, np.array([center_proj[0] * local_xyz[-1],
                                                            center_proj[1] * local_xyz[-1],
                                                            local_xyz[-1],
                                                            1.]))
                    local_xyz = local_xyz[:3]
                    # print(local_xyz)
                    # simply change x to -x is not enough
                    alpha = self._convertRot2Alpha(ry3d=ry, x3d=local_xyz[0], z3d=local_xyz[2])

                r11, r12, r13 = np.cos(ry), 0., np.sin(ry)
                r21, r22, r23 = 0., 1., 0.
                r31, r32, r33 = -np.sin(ry), 0., np.cos(ry)
                R = np.array([[r11, r12, r13],
                              [r21, r22, r23],
                              [r31, r32, r33]])
                proj_bbox3D, bbox3D8Points = self._get3DbboxProjection(projmat=P2, R=R, t=local_xyz, w=w, h=h, l=l)
                x1, x2 = np.min(proj_bbox3D[:, :, :, 0]), np.max(proj_bbox3D[:, :, :, 0])
                y1, y2 = np.min(proj_bbox3D[:, :, :, 1]), np.max(proj_bbox3D[:, :, :, 1])

                if np.all(bbox3D8Points[:, 2]) > 0:
                    objsInfo.append([category_name, x1, y1, x2, y2,
                                     center_proj,
                                     h, w, l, local_xyz,
                                     alpha, ry, bbox3D8Points, cad_index])
                    depth.append(local_xyz[-1])
        objsInfo = np.array(objsInfo, dtype=object)
        depth = np.array(depth)
        if len(objsInfo) > 0:
            objsInfo = objsInfo[np.argsort(depth)]
            # np.random.shuffle(objsInfo)
            return np.array(objsInfo)
        else:
            return np.array([])

    def _getP2AndP2Inv(self, calib_path):
        with open(calib_path) as fp:
            calib = fp.readlines()
        P2 = None
        for calib_line in calib:
            if calib_line.split(' ')[0] == 'P2:':
                P2 = np.array(calib_line.split(' ')[1:]).astype('float64')
        # P2 = np.array(calib[2].split(' ')[1:]).astype('float64')
        # print(calib[2].split(' ')[0])
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
        return projection_mat, projection_mat_inv

    def _getOffset(self, batchSize):
        offsetX = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[0])] * self._gridSize[1] * self._predNumPerGrid)
            , (self._predNumPerGrid, self._gridSize[1], self._gridSize[0])
        ), (1, 2, 0))
        offsetX = np.tile(np.reshape(offsetX, (1, self._gridSize[1], self._gridSize[0], self._predNumPerGrid)),
                          [batchSize, 1, 1, 1])
        offsetY = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[1])] * self._gridSize[0] * self._predNumPerGrid)
            , (self._predNumPerGrid, self._gridSize[0], self._gridSize[1])
        ), (2, 1, 0))
        offsetY = np.tile(np.reshape(offsetY, (1, self._gridSize[1], self._gridSize[0], self._predNumPerGrid)),
                          [batchSize, 1, 1, 1])
        return offsetX.astype('float32'), offsetY.astype('float32')

    def get3DboxAverage(self, save_path):
        bbox3D_dim = []
        for i, KITTITrain in enumerate(self._filename_list):
            print(i, len(self._filename_list))
            annotpath = os.path.join(self._KITTIAnnotPath, KITTITrain + '.mat')
            imagepath = os.path.join(self._KITTIImagePath, KITTITrain + '.png')
            calibpath = os.path.join(self._KITTI_calib_path, KITTITrain + '.txt')
            imageRow, imageCol, _ = cv2.imread(imagepath).shape
            P2, P2_inv = self._getP2AndP2Inv(calibpath)
            objsInfo = self._objMat2Elem(annot_path=annotpath, P2=P2, P2_inv=P2_inv)
            for objInfo in objsInfo:
                category_name, colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, alpha, ry, bbox3D8Points, cad_idx = objInfo
                bbox3D_dim.append([l,h,w])
        bbox3D_dim_avr = np.mean(bbox3D_dim, axis=0)
        save_path = os.path.join(save_path, 'anchor_bbox3D.npy')
        print(bbox3D_dim_avr)
        np.save(save_path, bbox3D_dim_avr)

    def getKMeansDist(self, save_path, k=None, max_iter=5000):
        if k == None:
            k = self._predNumPerGrid
        z_list = []
        for i, KITTITrain in enumerate(self._filename_list):
            print(i, len(self._filename_list))
            annotpath = os.path.join(self._KITTIAnnotPath, KITTITrain + '.mat')
            imagepath = os.path.join(self._KITTIImagePath, KITTITrain + '.png')
            calibpath = os.path.join(self._KITTI_calib_path, KITTITrain + '.txt')
            imageRow, imageCol, _ = cv2.imread(imagepath).shape
            P2, P2_inv = self._getP2AndP2Inv(calibpath)
            objsInfo = self._objMat2Elem(annot_path=annotpath, P2=P2, P2_inv=P2_inv)
            for objInfo in objsInfo:
                # _, _, colMin, rowMin, colMax, rowMax, h, w, l, local_xyz, cadIndex, azimuth, elevation = objInfo
                # truncated, occluded, colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, cadIndex, ry = objInfo
                category_name, colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, alpha, ry, bbox3D8Points, cad_idx = objInfo
                z_list.append(local_xyz[2])
        z_list = np.array(z_list)
        print('simple Euclidian dist')
        kmeans = kmeans_dist(X=z_list)
        dist_centers, xtoc, dist = kmeans.kmeansSample(max_iter=max_iter, k=k)
        min_arg = np.argsort(dist_centers)
        print(dist_centers[min_arg])
        np.save(os.path.join(save_path, 'global_anchor_z_e.npy'), dist_centers[min_arg])

        print('square dist')
        kmeans = kmeans_dist(X=z_list)
        dist_centers, xtoc, dist = kmeans.kmeansSample(max_iter=max_iter, k=k, sqr_scale=True)
        min_arg = np.argsort(dist_centers)
        print(dist_centers[min_arg])
        np.save(os.path.join(save_path, 'global_anchor_z_s.npy'), dist_centers[min_arg])

        print('log-scale dist')
        kmeans = kmeans_dist(X=z_list)
        dist_centers, xtoc, dist = kmeans.kmeansSample(max_iter=max_iter, k=k, log_scale=True)
        min_arg = np.argsort(dist_centers)
        print(dist_centers[min_arg])
        np.save(os.path.join(save_path, 'global_anchor_z_l.npy'), dist_centers[min_arg])

    def getImageAverage(self, save_path):
        pixel_sum = np.zeros((3,))
        pixel_dsquare_sum = np.zeros((3,))
        pixel_num = 0.
        for i, KITTITrain in enumerate(self._filename_list):
            print('mean', i, len(self._filename_list))
            imagepath = os.path.join(self._KITTIImagePath, KITTITrain + '.png')
            image = np.array(cv2.imread(imagepath))
            imgRow, imgCol, ch = image.shape
            pixel_num += imgRow*imgCol
            pixel_sum_curr = np.sum(np.sum(image, axis=0), axis=0)
            pixel_sum += pixel_sum_curr
        pixel_mean = pixel_sum / pixel_num
        for i, KITTITrain in enumerate(self._filename_list):
            print('std', i, len(self._filename_list))
            imagepath = os.path.join(self._KITTIImagePath, KITTITrain + '.png')
            image = np.array(cv2.imread(imagepath))
            pixel_dsquare_sum_curr = np.sum(np.sum(np.square(image - pixel_mean), axis=0), axis=0)
            pixel_dsquare_sum += pixel_dsquare_sum_curr
        pixel_std = np.sqrt(pixel_dsquare_sum / pixel_num)
        print(pixel_mean)
        print(pixel_std)
        np.save(os.path.join(save_path, 'pixel_mean.npy'), pixel_mean)
        np.save(os.path.join(save_path, 'pixel_std.npy'), pixel_std)

    def loadPrior(self):
        self._anchor_z_global = np.load(os.path.join(self._KITTI_anchor_path, 'global_anchor_z_s.npy'))
        self._anchor_bbox3D = np.load(os.path.join(self._KITTI_anchor_path, 'anchor_bbox3D.npy'))
        self._pixelMean = np.load(os.path.join(self._KITTI_anchor_path, 'pixel_mean.npy'))
        self._pixelStd = np.load(os.path.join(self._KITTI_anchor_path, 'pixel_std.npy'))


    def getNextBatch(self, batchSize=12, imageSize=None, gridSize=None, augmentation=True, dist_type='s'):
        if imageSize!=None:
            self._imageSize = imageSize
        if gridSize!=None:
            self._gridSize = gridSize
        inputImages, objnessImages, objnessCarImages, bbox2DDimImages, bbox2DXYImages,\
        bbox3DDimImages, localXYZImages, eulerRadImages, alphaImages = [], [], [], [], [], [], [], [], []
        bbox3D8PointsImages = []
        imageSizeList, P2List, P2InvList = [], [], []
        categoryList, outputImages, carInstList = [], [], []
        while len(inputImages) < batchSize:
            for filename in self._filename_list[self.dataStart:]:
                if len(inputImages) >= batchSize:
                    break
                is_flip = False
                if augmentation:
                    is_flip = np.random.rand() > 0.5

                calibpath = os.path.join(self._KITTI_calib_path, filename + '.txt')
                imagepath = os.path.join(self._KITTIImagePath, filename + '.png')
                annotpath = os.path.join(self._KITTIAnnotPath, filename + '.mat')

                P2, P2_inv = self._getP2AndP2Inv(calibpath)

                image = cv2.imread(imagepath, cv2.IMREAD_COLOR)
                image_row_org, image_col_org, _ = image.shape
                image = cv2.resize(image, self._imageSize)
                # if augmentation:
                    # if np.random.rand()>0.5:
                    #     image = imgAug(inputImage=image,
                    #                    crop=False, flip=False,
                    #                    gaussianBlur=True, channelInvert=True, brightness=True, hueSat=True)
                if is_flip:
                    image = cv2.flip(image, flipCode=1)
                image = (image - self._pixelMean)/self._pixelStd

                objlist = self._objMat2Elem(annot_path=annotpath, P2=P2, P2_inv=P2_inv, isFlip=is_flip)
                imageSizeImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 2])
                P2Image = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 4, 4])
                P2InvImage = np.zeros_like(P2Image)
                imageSizeImage[:,:,:,:] = image_col_org, image_row_org
                P2Image[:,:,:,:,:] = P2
                P2InvImage[:,:,:,:,:] = P2_inv

                objOrderingImage = -1 * np.ones([self._gridSize[1], self._gridSize[0], self._predNumPerGrid])
                carOrderingImage = -1 * np.ones([self._gridSize[1], self._gridSize[0], self._predNumPerGrid])
                objnessImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                objnessCarImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                bbox2DDimImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 4])
                bbox2DXYImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 2])
                bbox3DDimImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                localXYZImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                eulerRadImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                alphaImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                bbox3D8PointsImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 8, 3])
                categoryPerImage, outputPerImage, carInstPerImage = [], [], []
                itemIndex = 0
                carIndex = 0
                for objIndex, objInfo in enumerate(objlist):
                    category_name, colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, alpha, ry, bbox3D8Points, cad_idx = objInfo
                    colCenter = center_proj[0]
                    rowCenter = center_proj[1]

                    # color = (0, 255, 0)
                    # thickness = 2
                    # p0 = (int(colMin), int(rowMin))
                    # p1 = (int(colMax), int(rowMax))
                    # # print(p0)
                    # # print(p1)
                    # # print(image_bbox2D.shape)
                    # cv2.rectangle(img=image, pt1=p0, pt2=p1, color=color, thickness=thickness)
                    # cv2.circle(image, (colCenter, rowCenter), 5, (0,0,255), -1)

                    rowCenterOnGrid = rowCenter * self._gridSize[1] / image_row_org
                    colCenterOnGrid = colCenter * self._gridSize[0] / image_col_org
                    rowIndexOnGrid = int(rowCenterOnGrid)
                    colIndexOnGrid = int(colCenterOnGrid)
                    # rowIndexOnGrid = min(max(int(rowCenterOnGrid), 0), self._gridSize[1]-1)
                    # colIndexOnGrid = min(max(int(colCenterOnGrid), 0), self._gridSize[0]-1)
                    dx, dy = colCenterOnGrid - colIndexOnGrid, rowCenterOnGrid - rowIndexOnGrid

                    if rowIndexOnGrid >= 0 and rowIndexOnGrid < self._gridSize[1] \
                            and colIndexOnGrid >= 0 and colIndexOnGrid < self._gridSize[0]:
                        is_assigned = False
                        if dist_type == 's':
                            dist_diff = np.abs(np.square(self._anchor_z_global) - np.square(local_xyz[-1]))
                        elif dist_type == 'e':
                            dist_diff = np.abs(self._anchor_z_global - local_xyz[-1])
                        elif dist_type == 'l':
                            dist_diff = np.abs(np.log(self._anchor_z_global) - np.log(local_xyz[-1]))
                        else:
                            print('dist type : e')
                            dist_diff = np.abs(self._anchor_z_global - local_xyz[-1])
                        predIndex = np.argmin(dist_diff)
                        while is_assigned==False and predIndex<self._predNumPerGrid:
                            # print(anchor_z_per_grid)
                            # print(dist_diff)
                            # print(predIndex)
                            if objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex, 0] == 0:
                                objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex, 0] = 1
                                bbox2DDimImage[rowIndexOnGrid, colIndexOnGrid, predIndex,:] = colMin / image_col_org, rowMin / image_row_org, colMax / image_col_org, rowMax / image_row_org
                                bbox2DXYImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = dx, dy
                                bbox3DDimImage[rowIndexOnGrid, colIndexOnGrid, predIndex,:] = l, h, w  # kitti car coordinate
                                localXYZImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = local_xyz
                                # print(h,w,l)
                                alphaImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = alpha
                                eulerRadImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = ry
                                bbox3D8PointsImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :, :] = bbox3D8Points
                                # set item order
                                objOrderingImage[rowIndexOnGrid, colIndexOnGrid, predIndex] = itemIndex
                                categoryVector = np.zeros(category_num)
                                categoryVector[category_to_index[category_name]] = 1
                                categoryPerImage.append(categoryVector)

                                itemIndex += 1
                                is_assigned = True

                                if category_name == 'Car':
                                    objnessCarImage[rowIndexOnGrid, colIndexOnGrid, predIndex, 0] = 1
                                    carOrderingImage[rowIndexOnGrid, colIndexOnGrid, predIndex] = carIndex
                                    carIndex += 1
                                    # car instance vector
                                    carInstVector = np.zeros(len(self._KITTI3DShapes))
                                    carInstVector[cad_idx - 1] = 1
                                    carInstPerImage.append(carInstVector)
                                    # car 3D CAD model
                                    car3DCAD = self._KITTI3DShapes[cad_idx - 1]
                                    outputPerImage.append(car3DCAD)

                            else:
                                # print('something already exist :', rowIndexOnGrid, colIndexOnGrid, predIndex)
                                predIndex += 1
                            # break # just escape - do not assign if predictor is already occupied

                if itemIndex > 0:
                    # if True:
                    # cv2.imwrite('test/' + str(len(inputImages)) + '.png', inputImage)
                    # image = cv2.resize(image, self._imageSize)
                    inputImages.append(image)
                    objnessImages.append(objnessImage)
                    objnessCarImages.append(objnessCarImage)
                    bbox2DDimImages.append(bbox2DDimImage)
                    bbox2DXYImages.append(bbox2DXYImage)
                    bbox3DDimImages.append(bbox3DDimImage)
                    localXYZImages.append(localXYZImage)
                    eulerRadImages.append(eulerRadImage)
                    alphaImages.append(alphaImage)
                    bbox3D8PointsImages.append(bbox3D8PointsImage)

                    # imageSizeList.append([imageRowOrg, imageColOrg])
                    # P2List.append(projection_mat)
                    # P2InvList.append(projection_mat_inv)
                    imageSizeList.append(imageSizeImage)
                    P2List.append(P2Image)
                    P2InvList.append(P2InvImage)

                    for gridRow in range(self._gridSize[1]):
                        for gridCol in range(self._gridSize[0]):
                            for predIndex in range(self._predNumPerGrid):
                                objOrder = int(objOrderingImage[gridRow, gridCol, predIndex])
                                carOrder = int(carOrderingImage[gridRow, gridCol, predIndex])
                                if objOrder >= 0:
                                    categoryList.append(categoryPerImage[objOrder])
                                if carOrder >= 0:
                                    outputImages.append(outputPerImage[carOrder])
                                    carInstList.append(carInstPerImage[carOrder])
                # except:
                #     pass
                self.dataStart += 1
                if self.dataStart >= self.dataLength:  # out of dataset length
                    self.epoch += 1
                    self._dataShuffle()
                    break
        inputImages = np.array(inputImages).astype('float32')
        objnessImages = np.array(objnessImages).astype('float32')
        objnessCarImages = np.array(objnessCarImages).astype('float32')
        bbox2DDimImages = np.array(bbox2DDimImages).astype('float32')
        bbox2DXYImages = np.array(bbox2DXYImages).astype('float32')
        bbox3DDimImages = np.array(bbox3DDimImages).astype('float32')
        localXYZImages = np.array(localXYZImages).astype('float32')
        eulerRadImages = np.array(eulerRadImages).astype('float32')
        alphaImages = np.array(alphaImages).astype('float32')
        bbox3D8PointsImages = np.array(bbox3D8PointsImages).astype('float32')
        # print(bbox3D8PointsImages.shape)
        imageSizeList = np.array(imageSizeList).astype('float32')
        P2List = np.array(P2List).astype('float32')
        P2InvList = np.array(P2InvList).astype('float32')
        categoryList = np.array(categoryList).astype('float32')
        outputImages = np.array(outputImages).astype('float32')
        carInstList = np.array(carInstList).astype('float32')

        offsetX, offsetY = self._getOffset(batchSize=len(inputImages))
        anchor_z_global = self._anchor_z_global.astype('float32')
        anchor_bbox3D = self._anchor_bbox3D.astype('float32')
        # anchor_z_global_grid = np.zeros([batchSize, self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
        # anchor_bbox3D_grid = np.zeros([batchSize, self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
        # anchor_z_global_grid[:,:,:,:,:] = anchor_z_global
        # anchor_bbox3D_grid[:,:,:,:,:] = anchor_bbox3D

        return offsetX, offsetY, inputImages, objnessImages, objnessCarImages, \
        bbox2DDimImages, bbox2DXYImages, bbox3DDimImages, localXYZImages, alphaImages, \
        bbox3D8PointsImages, \
        imageSizeList, P2List, P2InvList, categoryList, \
        outputImages, carInstList, \
        anchor_z_global, anchor_bbox3D

if __name__ == '__main__':
    data_path = '/home/yonsei/dataset/kitti/'
    kitti = dataLoader(
        KITTIImagePath=os.path.join(data_path, 'data_object_image_2/training/image_2/'),
        KITTIAnnotPath=os.path.join(data_path, '3DVP_Annotations/Annotations/'),
        KITTI_calib_path=os.path.join(data_path, 'data_object_calib/training/calib/'),
        KITTI3DShapePath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/CAD/car/',
        KITTI_anchor_path=''
    )
    kitti.get3DboxAverage(save_path=data_path)
    kitti.getKMeansDist(k=12, max_iter=1000, save_path=data_path)
    kitti.getImageAverage(save_path=data_path)

    # for data_type in ['kitti_split1', 'kitti_split2']:
    #     data_path = os.path.join('/home/yonsei/pyws/anchor_dist_kitti/data/', data_type)
    #     kitti = dataLoader(
    #         KITTIImagePath=os.path.join(data_path, 'training/image_2/'),
    #         KITTIAnnotPath=os.path.join(data_path, 'training/3dv_2/'),
    #         KITTI_calib_path=os.path.join(data_path, 'training/calib/'),
    #         KITTI3DShapePath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/CAD/car/',
    #         KITTI_anchor_path=''
    #     )
    #     kitti.get3DboxAverage(save_path=data_path)
    #     kitti.getKMeansDist(k=12, max_iter=1000, save_path=data_path)
    #     kitti.getImageAverage(save_path=data_path)





















