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

class dataLoader(object):
    def __init__(self,
                 imageSize=(1280, 384),
                 gridSize=(40, 12),
                 predNumPerGrid=5,
                 KITTIAnnotPath='/home/yonsei/dataset/KITTI/3DVP_Annotations/Annotations/',
                 KITTIImagePath='/home/yonsei/dataset/KITTI/data_object_image_2/training/image_2/',
                 KITTI_calib_path='/home/yonsei/dataset/KITTI/data_object_calib/training/calib/',
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
        self._KITTI_anchor_path = KITTI_anchor_path
        self._isTrain = is_train
        self._filename_list = np.sort(os.listdir(self._KITTIAnnotPath))
        for i, filename in enumerate(self._filename_list):
            self._filename_list[i] = filename.split('.')[0]
        self.dataLength = len(self._filename_list)

        print('set kitti dataset...')
        if self._isTrain:
            self._dataShuffle()

    def _dataShuffle(self):
        # print('data path shuffle...')
        self.dataStart = 0
        np.random.shuffle(self._filename_list)
        # print('done! : ' + str(self.dataLength))

    # def _load3DShapes(self):
    #     print('load 3d shapes for KITTI...')
    #     self._KITTI3DShapes = []
    #     CADModelList = os.listdir(self._KITTI3DShapePath)
    #     CADModelList.sort()
    #     # print CADModelList
    #     for CADModel in CADModelList:
    #         if CADModel.split(".")[-1] == 'npy':
    #             shape = np.load(os.path.join(self._KITTI3DShapePath, CADModel)).reshape(64, 64, 64, 1)
    #             self._KITTI3DShapes.append(shape)
    #     self._KITTI3DShapes = np.array(self._KITTI3DShapes)
    #     self._KITTI3DShapes = np.where(self._KITTI3DShapes>0, 1.0, 0.0)
    #     print('done!')

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

    def _obj2Elem(self, annot_path, P2, P2_inv, image_size, isFlip=False):
        image_col, image_row = image_size
        with open(annot_path) as fp:
            objs = fp.readlines()
        objsInfo = []
        depth = []
        for obj in objs:
            obj = obj.split(' ')
            cls = obj[0]
            truncated = float(obj[1])
            occluded = float(obj[2])
            alpha = float(obj[3])
            x1, y1, x2, y2 = float(obj[4]), float(obj[5]), float(obj[6]), float(obj[7])
            h, w, l = float(obj[8]), float(obj[9]), float(obj[10])
            local_xyz = np.array([float(obj[11]), float(obj[12]), float(obj[13])])
            ry = float(obj[14])
            alpha = self._convertRot2Alpha(ry3d=ry, x3d=local_xyz[0], z3d=local_xyz[2])
            if cls == 'Car':
                local_xyz[1] = local_xyz[1] - h / 2.  # 3d bbox center is object center, not bottom

                center_proj = np.array([local_xyz[0], local_xyz[1], local_xyz[2], 1.]).astype('float128')
                center_proj = matmul4x1(P2, center_proj)
                center_proj = center_proj[:2] / center_proj[2]

                if isFlip:
                    # need to change the orientation
                    if ry>0:
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
                    objsInfo.append([x1, y1, x2, y2,
                                     center_proj,
                                     h, w, l, local_xyz,
                                     alpha, ry, bbox3D8Points])
                    depth.append(local_xyz[-1])
        objsInfo = np.array(objsInfo, dtype=object)
        depth = np.array(depth)
        if len(objsInfo)>0:
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
            annotpath = os.path.join(self._KITTIAnnotPath, KITTITrain + '.txt')
            imagepath = os.path.join(self._KITTIImagePath, KITTITrain + '.png')
            calibpath = os.path.join(self._KITTI_calib_path, KITTITrain + '.txt')
            imageRow, imageCol, _ = cv2.imread(imagepath).shape
            P2, P2_inv = self._getP2AndP2Inv(calibpath)
            objsInfo = self._obj2Elem(annot_path=annotpath, P2=P2, P2_inv=P2_inv,
                                      image_size=[imageCol, imageRow])
            for objInfo in objsInfo:
                colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, alpha, ry, bbox3D8Points = objInfo
                bbox3D_dim.append([l,h,w])
        bbox3D_dim_avr = np.mean(bbox3D_dim, axis=0)
        save_path = os.path.join(save_path, 'anchor_bbox3D.npy')
        np.save(save_path, bbox3D_dim_avr)

    def getKMeansDist(self, save_path, k=None, max_iter=5000):
        if k == None:
            k = self._predNumPerGrid
        z_list = []
        for i, KITTITrain in enumerate(self._filename_list):
            print(i, len(self._filename_list))
            annotpath = os.path.join(self._KITTIAnnotPath, KITTITrain + '.txt')
            imagepath = os.path.join(self._KITTIImagePath, KITTITrain + '.png')
            calibpath = os.path.join(self._KITTI_calib_path, KITTITrain + '.txt')
            imageRow, imageCol, _ = cv2.imread(imagepath).shape
            P2, P2_inv = self._getP2AndP2Inv(calibpath)
            objsInfo = self._obj2Elem(annot_path=annotpath, P2=P2, P2_inv=P2_inv,
                                      image_size=[imageCol, imageRow])
            for objInfo in objsInfo:
                # _, _, colMin, rowMin, colMax, rowMax, h, w, l, local_xyz, cadIndex, azimuth, elevation = objInfo
                # truncated, occluded, colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, cadIndex, ry = objInfo
                colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, alpha, ry, bbox3D8Points = objInfo
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


    def getNextBatch(self, batchSize=12, imageSize=None, gridSize=None, augmentation=True, dist_type='s'):
        if imageSize!=None:
            self._imageSize = imageSize
        if gridSize!=None:
            self._gridSize = gridSize
        inputImages, objnessImages, bbox2DDimImages, bbox2DXYImages,\
        bbox3DDimImages, localXYZImages, eulerRadImages, alphaImages = [], [], [], [], [], [], [], []
        bbox3D8PointsImages = []
        imageSizeList, P2List, P2InvList = [], [], []
        anchor_z_global = np.load(os.path.join(self._KITTI_anchor_path, 'global_anchor_z_'+dist_type+'.npy'))
        anchor_bbox3D = np.load(os.path.join(self._KITTI_anchor_path, 'anchor_bbox3D.npy'))
        while len(inputImages) < batchSize:
            for filename in self._filename_list[self.dataStart:]:
                if len(inputImages) >= batchSize:
                    break
                is_flip = False
                if augmentation:
                    is_flip = np.random.rand() > 0.5

                calibpath = os.path.join(self._KITTI_calib_path, filename + '.txt')
                imagepath = os.path.join(self._KITTIImagePath, filename + '.png')
                annotpath = os.path.join(self._KITTIAnnotPath, filename + '.txt')

                P2, P2_inv = self._getP2AndP2Inv(calibpath)

                image = cv2.imread(imagepath, cv2.IMREAD_COLOR)
                image_row_org, image_col_org, _ = image.shape
                image = cv2.resize(image, self._imageSize)
                # if augmentation:
                #     if np.random.rand()>0.5:
                # image = imgAug(inputImage=image,
                #                crop=False, flip=False,
                #                gaussianBlur=True, channelInvert=True, brightness=True, hueSat=True)
                if is_flip:
                    image = cv2.flip(image, flipCode=1)

                objlist = self._obj2Elem(annot_path=annotpath, P2=P2, P2_inv=P2_inv,
                                         image_size=[image_col_org, image_row_org], isFlip=is_flip)

                imageSizeImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 2])
                P2Image = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 4, 4])
                P2InvImage = np.zeros_like(P2Image)
                imageSizeImage[:,:,:,:] = image_col_org, image_row_org
                P2Image[:,:,:,:,:] = P2
                P2InvImage[:,:,:,:,:] = P2_inv

                objOrderingImage = -1 * np.ones([self._gridSize[1], self._gridSize[0], self._predNumPerGrid])
                objnessImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                bbox2DDimImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 4])
                bbox2DXYImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 2])
                bbox3DDimImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                localXYZImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                eulerRadImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                alphaImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                bbox3D8PointsImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 8, 3])
                itemIndex = 0

                for objIndex, objInfo in enumerate(objlist):
                    colMin, rowMin, colMax, rowMax, center_proj, h, w, l, local_xyz, alpha, ry, bbox3D8Points = objInfo
                    colCenter = center_proj[0]
                    rowCenter = center_proj[1]

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
                            dist_diff = np.abs(np.square(anchor_z_global) - np.square(local_xyz[-1]))
                        elif dist_type == 'e':
                            dist_diff = np.abs(anchor_z_global - local_xyz[-1])
                        elif dist_type == 'l':
                            dist_diff = np.abs(np.log(anchor_z_global) - np.log(local_xyz[-1]))
                        else:
                            print('dist type : e')
                            dist_diff = np.abs(anchor_z_global - local_xyz[-1])
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
                                itemIndex += 1
                                is_assigned = True
                            else:
                                # print('something already exist :', rowIndexOnGrid, colIndexOnGrid, predIndex)
                                predIndex += 1
                            # break # just escape - do not assign if predictor is already occupied

                if itemIndex > 0:
                    # if True:
                    # cv2.imwrite('test/' + str(len(inputImages)) + '.png', inputImage)
                    inputImages.append(image)
                    objnessImages.append(objnessImage)
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

                    # for gridRow in range(self._gridSize[1]):
                    #     for gridCol in range(self._gridSize[0]):
                    #         for predIndex in range(self._predNumPerGrid):
                    #             objOrder = int(objOrderingImage[gridRow, gridCol, predIndex])
                    #             if objOrder >= 0:
                    #                 outputImages.append(outputPerImage[objOrder])
                    #                 instList.append(instPerImage[objOrder])
                    #                 eulerList.append(EulerPerImage[objOrder])
                # except:
                #     pass
                self.dataStart += 1
                if self.dataStart >= self.dataLength:  # out of dataset length
                    self.epoch += 1
                    self._dataShuffle()
                    break
        inputImages = np.array(inputImages).astype('float32')
        objnessImages = np.array(objnessImages).astype('float32')
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
        offsetX, offsetY = self._getOffset(batchSize=len(inputImages))
        anchor_z_global = anchor_z_global.astype('float32')
        anchor_bbox3D = anchor_bbox3D.astype('float32')
        # anchor_z_global_grid = np.zeros([batchSize, self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
        # anchor_bbox3D_grid = np.zeros([batchSize, self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
        # anchor_z_global_grid[:,:,:,:,:] = anchor_z_global
        # anchor_bbox3D_grid[:,:,:,:,:] = anchor_bbox3D

        return offsetX, offsetY, inputImages, objnessImages,\
    bbox2DDimImages, bbox2DXYImages, bbox3DDimImages, localXYZImages, alphaImages, \
    bbox3D8PointsImages, \
    imageSizeList, P2List, P2InvList,\
    anchor_z_global, anchor_bbox3D

if __name__ == '__main__':
    for data_type in ['kitti_split1_added_full', 'kitti_split1_org', 'kitti_split2_added_full', 'kitti_split2_org']:
        data_path = os.path.join('/home/yonsei/pyws/NOLBO_grid_anchor_no3Ddec/NOLBO_2D_3D_IoU_exp_viewing/data/', data_type)
        kitti = dataLoader(
            KITTIImagePath=os.path.join(data_path, 'training/image_2/'),
            KITTIAnnotPath=os.path.join(data_path, 'training/label_2/'),
            KITTI_calib_path=os.path.join(data_path, 'training/calib/'),
            KITTI_anchor_path=''
        )
        kitti.get3DboxAverage(save_path=data_path)
        kitti.getKMeansDist(k=12, max_iter=1000, save_path=data_path)
























