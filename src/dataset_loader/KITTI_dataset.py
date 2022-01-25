import scipy.io
import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from src.module.function import kmeans_dist
from src.dataset_loader.datasetUtils import *
# from utils import imageAugmentation
# from utils import voxelBlur
#
#
# class KITTISingleObject(object):
#     def __init__(self,
#                  imageSize=(1280, 384),
#                  KITTIAnnotPath='/media/yonsei/4TB_HDD/downloads/KITTI/3DVP_Annotations/Annotations/',
#                  KITTIDataListPath='/media/yonsei/4TB_HDD/downloads/KITTI/trainvalsplit_3DOP_MONO3D/',
#                  KITTIImagePath='/media/yonsei/4TB_HDD/downloads/KITTI/data_object_image_2/training/image_2/',
#                  KITTI_calib_path='/media/yonsei/4TB_HDD/downloads/KITTI/data_object_calib/training/calib/',
#                  KITTI3DShapePath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/CAD/car/',
#                  trainOrVal='train',
#                  convertToPascal = True,
#                  ):
#         self.dataStart = 0
#         self.dataLength = 0
#         self.epoch = 0
#
#         self._imageSize = imageSize
#         self._KITTIAnnotPath = KITTIAnnotPath
#
#         self._isTrain = True
#         self._trainOrVal = trainOrVal
#         if self._trainOrVal == 'train':
#             KITTIDataListPath = os.path.join(KITTIDataListPath, 'train.txt')
#             self._isTrain = True
#         elif self._trainOrVal == 'val':
#             KITTIDataListPath = os.path.join(KITTIDataListPath, 'val.txt')
#             self._isTrain = False
#         else:
#             print('trainOrVal : train or val')
#             return
#         self._KITTIDataListPath4Eval = np.loadtxt(KITTIDataListPath, dtype='str')
#         self._KITTIDataListPath = np.loadtxt(KITTIDataListPath, dtype='str')
#
#         self._KITTIImagePath = KITTIImagePath
#         self._KITTI3DShapePath = KITTI3DShapePath
#         self._convertTopascal=convertToPascal
#         self._KITTI3DShapes = None
#
#         print('set kitti dataset...')
#         self._load3DShapes()
#         self._dataShuffle()
#
#     def _dataShuffle(self):
#         print('')
#         print('KITTI data path shuffle...')
#         self.dataStart = 0
#         np.random.shuffle(self._KITTIDataListPath)
#         self.dataLength = len(self._KITTIDataListPath)
#         print('done! : ' + str(self.dataLength))
#
#     def _load3DShapes(self):
#         print('load 3d shapes for KITTI...')
#         self._KITTI3DShapes = []
#         self._KITTI3DShapesBlur = []
#         CADModelList = os.listdir(self._KITTI3DShapePath)
#         CADModelList.sort()
#         # print CADModelList
#         for CADModel in CADModelList:
#             if CADModel.split(".")[-1] == 'npy':
#                 shape = np.load(os.path.join(self._KITTI3DShapePath, CADModel)).reshape(64, 64, 64, 1)
#                 self._KITTI3DShapes.append(shape)
#                 self._KITTI3DShapesBlur.append(voxelBlur(shape))
#         self._KITTI3DShapes = np.array(self._KITTI3DShapes)
#         self._KITTI3DShapes = np.where(self._KITTI3DShapes>0, 1.0, 0.0)
#         self._KITTI3DShapesBlur = np.array(self._KITTI3DShapesBlur)
#         print('done!')
#
#     def _objMat2Elem(self, objMat):
#         objs = objMat['record']['objects'][0][0][0]
#         objsInfo = []
#         for obj in objs:
#             # print obj[0][0]
#             if obj[0][0] == 'Car':
#                 truncated = obj[1][0][0]
#                 occluded = obj[2][0][0]
#                 x1, y1, x2, y2 = obj[4][0][0], obj[5][0][0], obj[6][0][0], obj[7][0][0]
#                 cad_index = int(obj[13][0][0]) # [1,2,3,4,5,6,7] -> [1,2,3,4,5,6,9]
#                 if cad_index == 7:
#                     cad_index = 9
#                 azimuth, elevation = obj[14][0][0], obj[15][0][0]
#                 objsInfo.append([truncated, occluded, x1, y1, x2, y2, cad_index, azimuth, elevation])
#         objsInfo = np.array(objsInfo)
#         if len(objsInfo)>0:
#             # objsInfo = objsInfo[objsInfo[:, 1].argsort()]
#             np.random.shuffle(objsInfo)
#             return np.array(objsInfo)
#         else:
#             return np.array([])
#
#     def getNextBatch(self, batchSizeof3DShape=32, imageSize=None):
#         if imageSize!=None:
#             self._imageSize = imageSize
#         inputImages = []
#         outputImages, instList, EulerRadList = [], [], []
#         outputImagesBlur = []
#         while len(outputImages)==0:
#             for KITTITrain in self._KITTIDataListPath[self.dataStart:]:
#                 try:
#                     objMat = scipy.io.loadmat(os.path.join(self._KITTIAnnotPath, KITTITrain + '.mat'))
#                     objsInfo = self._objMat2Elem(objMat)
#                     if len(objsInfo)>0:
#                         if len(outputImages) > batchSizeof3DShape and len(inputImages)>0:
#                             break
#                         inputImage = cv2.imread(os.path.join(self._KITTIImagePath, KITTITrain + '.png'), cv2.IMREAD_COLOR)
#                         for objIndex, objInfo in enumerate(objsInfo):
#                             _, _, colMin, rowMin, colMax, rowMax, cadIndex, azimuth, elevation = objInfo
#                             colMin, colMax, rowMin, rowMax = int(float(colMin)), int(float(colMax)), int(float(rowMin)), int(float(rowMax))
#                             inputImageCurr = inputImage[rowMin:rowMax, colMin:colMax]
#                             inputImageCurr = cv2.resize(inputImageCurr, dsize=self._imageSize, interpolation=cv2.INTER_CUBIC)
#                             if np.random.rand() < 0.9:
#                                 inputImageCurr = imageAugmentation(inputImage=inputImageCurr, crop=True, flip=False, gaussianBlur=True)
#                             cadIndex = int(cadIndex)
#                             carInstVector = np.zeros(len(self._KITTI3DShapes))
#                             carInstVector[cadIndex - 1] = 1
#                             car3DCAD = self._KITTI3DShapes[cadIndex - 1]
#                             car3DCADBlur = self._KITTI3DShapesBlur[cadIndex - 1]
#                             if self._convertTopascal==True:
#                                 azimuth,elevation,inPlaneRot = (-float(azimuth))/180.0*np.pi, -float(elevation)/180.0*np.pi, 0.0
#                             else:
#                                 azimuth, elevation, inPlaneRot = -float(azimuth)/180.0*np.pi, -float(elevation)/180.0*np.pi, 0.0
#                             EulerRad = np.array([azimuth, elevation, inPlaneRot])
#
#                             inputImages.append(inputImageCurr)
#                             outputImages.append(car3DCAD)
#                             outputImagesBlur.append(car3DCADBlur)
#                             instList.append(carInstVector)
#                             EulerRadList.append(EulerRad)
#                 except:
#                     pass
#                 self.dataStart += 1
#                 if self.dataStart >= self.dataLength:  # out of dataset length
#                     self.epoch += 1
#                     self._dataShuffle()
#                     break
#         inputImages = np.array(inputImages).astype('float')
#         outputImages = np.array(outputImages).astype('float')
#         outputImagesBlur = np.array(outputImages).astype('float')
#         instList = np.array(instList).astype('float')
#         EulerRadList = np.array(EulerRadList).astype('float')
#         # print(inputImages.shape)
#         # print(instList.shape)
#
#         batchDict = {
#             'inputImages': inputImages,
#             'outputImages': outputImages,
#             'outputImagesBlur': outputImagesBlur,
#             'instList': instList,
#             'sin' : np.sin(EulerRadList),
#             'cos' : np.cos(EulerRadList),
#         }
#         return batchDict

class dataLoader(object):
    def __init__(self,
                 imageSize=(1280, 384),
                 gridSize=(40, 12),
                 predNumPerGrid=5,
                 KITTIAnnotPath='/home/yonsei/dataset/KITTI/3DVP_Annotations/Annotations/',
                 KITTIDataListPath='/home/yonsei/dataset/KITTI/trainvalsplit_3DOP_MONO3D/',
                 KITTIImagePath='/home/yonsei/dataset/KITTI/data_object_image_2/training/image_2/',
                 KITTI_calib_path='/home/yonsei/dataset/KITTI/data_object_calib/training/calib/',
                 KITTI3DShapePath='/home/yonsei/dataset/PASCAL3D+_release1.1/CAD/car/',
                 trainOrVal='train',
                 ):
        self.dataStart = 0
        self.dataLength = 0
        self.epoch = 0

        self._imageSize = imageSize
        self._gridSize = gridSize
        self._predNumPerGrid = predNumPerGrid
        self._KITTIAnnotPath = KITTIAnnotPath

        self._isTrain = True
        self._trainOrVal = trainOrVal
        if self._trainOrVal == 'train':
            trainPath = os.path.join(KITTIDataListPath, 'train.txt')
            self._KITTIDataListPath = np.loadtxt(trainPath, dtype='str')
            self._isTrain = True
        elif self._trainOrVal == 'val':
            valPath = os.path.join(KITTIDataListPath, 'val.txt')
            self._KITTIDataListPath = np.loadtxt(valPath, dtype='str')
            self._isTrain = False
        elif self._trainOrVal == 'trainval':
            trainPath = os.path.join(KITTIDataListPath, 'train.txt')
            valPath = os.path.join(KITTIDataListPath, 'val.txt')
            trainDataListPath = np.loadtxt(trainPath, dtype='str')
            valDataListPath = np.loadtxt(valPath, dtype='str')
            self._KITTIDataListPath = np.concatenate([trainDataListPath, valDataListPath], axis=0)
            self._isTrain = True
        else:
            print('trainOrVal : train or val')
            return
        # self._KITTIDataListPath4Eval = np.loadtxt(KITTIDataListPath, dtype='str')
        # self._KITTIDataListPath = np.loadtxt(KITTIDataListPath, dtype='str')

        self._KITTIImagePath = KITTIImagePath
        self._KITTI_calib_path = KITTI_calib_path
        self._KITTI3DShapePath = KITTI3DShapePath
        self._KITTI3DShapes = None

        # Euclidian dist
        # self.anchor_boxes = np.array(
        #     [[0.07675386, 0.0372182],
        #     [0.24211247, 0.13785882],
        #     [0.38803145, 0.16910161],
        #     [0.14466664, 0.07742635],
        #     [0.50000369, 0.25940277]]
        # )

        # IoU dist
        self.anchor_boxes = np.array(
            [[0.30958266, 0.14906247],
             [0.4714302, 0.24787024],
             [0.1141037, 0.05666735],
             [0.06811035, 0.02973031],
             [0.17434579, 0.09866836]]
        )

        print('set kitti dataset...')
        self._load3DShapes()
        self._dataShuffle()

    def _dataShuffle(self):
        # print('')
        # print('data path shuffle...')
        self.dataStart = 0
        np.random.shuffle(self._KITTIDataListPath)
        self.dataLength = len(self._KITTIDataListPath)
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
        print('done!')

    def _objMat2Elem(self, objMat):
        objs = objMat['record']['objects'][0][0][0]
        objsInfo = []
        for obj in objs:
            # print obj[0][0]
            if obj[0][0] == 'Car':
                truncated = obj[1][0][0]
                occluded = obj[2][0][0]
                x1, y1, x2, y2 = obj[4][0][0], obj[5][0][0], obj[6][0][0], obj[7][0][0]
                h, w, l = obj[8][0][0], obj[9][0][0], obj[10][0][0]
                local_xyz = np.array(obj[11][0])
                ry = obj[12][0][0]
                cad_index = int(obj[13][0][0]) # [1,2,3,4,5,6,7] -> [1,2,3,4,5,6,9]
                if cad_index == 7:
                    cad_index = 9
                azimuth, elevation = obj[14][0][0], obj[15][0][0]
                objsInfo.append([truncated, occluded,
                                 x1, y1, x2, y2,
                                 h, w, l, local_xyz,
                                 cad_index, azimuth, elevation])
        objsInfo = np.array(objsInfo)
        if len(objsInfo)>0:
            # objsInfo = objsInfo[objsInfo[:, 1].argsort()]
            np.random.shuffle(objsInfo)
            return np.array(objsInfo)
        else:
            return np.array([])

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

    def getKMeansDist(self, k=None, max_iter=5000):
        '''
        bbox hw
        k=5
        [[0.07675386 0.0372182 ]
         [0.24211247 0.13785882]
         [0.38803145 0.16910161]
         [0.14466664 0.07742635]
         [0.50000369 0.25940277]]
        '''
        if k == None:
            k = self._predNumPerGrid
        bboxHW_normalized = []
        dist_norm = []

        # imageColMean, imageRowMean = 0., 0.
        # for i, KITTITrain in enumerate(self._KITTIDataListPath):
        #     print(i, len(self._KITTIDataListPath))
        #     objMat = scipy.io.loadmat(os.path.join(self._KITTIAnnotPath, KITTITrain + '.mat'))
        #     imageColOrg, imageRowOrg, _ = objMat['record']['imgsize'][0][0][0]
        #     imageColMean += imageColOrg
        #     imageRowMean += imageRowOrg
        # imageColMean, imageRowMean = imageColMean / len(self._KITTIDataListPath), imageRowMean / len(
        #     self._KITTIDataListPath)
        # print(imageColMean, imageRowMean)
        imageColMean, imageRowMean = 1239.9163213474135, 374.4770752573185

        for i, KITTITrain in enumerate(self._KITTIDataListPath):
            # if i == 300:
            #     break
            print(i, len(self._KITTIDataListPath))
            objMat = scipy.io.loadmat(os.path.join(self._KITTIAnnotPath, KITTITrain + '.mat'))
            imageColOrg, imageRowOrg, _ = objMat['record']['imgsize'][0][0][0]
            # print(imageColOrg, imageRowOrg)
            objsInfo = self._objMat2Elem(objMat)
            if len(objsInfo)>0:
                # inputImage = cv2.imread(os.path.join(self._KITTIImagePath, KITTITrain + '.png'), cv2.IMREAD_COLOR)
                # imageRowOrg, imageColOrg, _ = inputImage.shape
                for objInfo in objsInfo:
                    _, _, colMin, rowMin, colMax, rowMax, h, w, l, local_xyz, cadIndex, azimuth, elevation = objInfo
                    bbox_h = (rowMax-rowMin)/imageRowOrg * imageRowMean
                    bbox_w = (colMax-colMin)/imageColOrg * imageColMean
                    bboxHW_normalized.append([bbox_h, bbox_w])
                    d = np.sqrt(np.sum(np.square(local_xyz)))
                    dist_norm.append(d)
        bboxHW_normalized = np.array(bboxHW_normalized)
        dist_norm = np.array(dist_norm)
        print(bboxHW_normalized.shape)
        print(dist_norm.shape)

        def getBboxWithX2C(bboxhw_n, xtoc, k):
            bboxhw_means = np.zeros((k, 2))
            for jc in range(k):
                c = np.where(xtoc == jc)[0]
                if len(c) > 0:
                    # bboxhw_means[jc] = np.mean(bboxhw_n[c], axis=0)
                    h_weighted_sum = np.sum(bboxhw_n[c, 0] * np.square(bboxhw_n[c, 1])) / np.sum(np.square(bboxhw_n[c, 1]))
                    w_weighted_sum = np.sum(bboxhw_n[c, 1] * np.square(bboxhw_n[c, 0])) / np.sum(np.square(bboxhw_n[c, 0]))
                    bboxhw_means[jc] = h_weighted_sum, w_weighted_sum
            return bboxhw_means

        print('simple Euclidian dist')
        for i in [2,3,5,7,9]:
            kmeans = kmeans_dist(X=dist_norm)
            dist_centers, xtoc, dist = kmeans.kmeansSample(max_iter=max_iter, k=i)
            bbox_centers = getBboxWithX2C(bboxhw_n=bboxHW_normalized, xtoc=xtoc, k=i)
            min_arg = np.argsort(dist_centers)
            for k in range(i):
                print(len(np.where(xtoc==k)[0])/len(xtoc))
                print(np.var(dist[np.where(xtoc == k)[0]]))
            print(i)
            print(dist_centers[min_arg])
            print(bbox_centers[min_arg])

        print('square dist')
        for i in [2, 3, 5, 7, 9]:
            kmeans = kmeans_dist(X=dist_norm)
            dist_centers, xtoc, dist = kmeans.kmeansSample(max_iter=max_iter, k=i, sqr_scale=True)
            bbox_centers = getBboxWithX2C(bboxhw_n=bboxHW_normalized, xtoc=xtoc, k=i)
            min_arg = np.argsort(dist_centers)
            for k in range(i):
                print(len(np.where(xtoc == k)[0]) / len(xtoc))
                print(np.var(dist[np.where(xtoc == k)[0]]))
            print(i)
            print(dist_centers[min_arg])
            print(bbox_centers[min_arg])

        print('log-scale dist')
        for i in [2, 3, 5, 7, 9]:
            kmeans = kmeans_dist(X=dist_norm)
            dist_centers, xtoc, dist = kmeans.kmeansSample(max_iter=max_iter, k=i, log_scale=True)
            bbox_centers = getBboxWithX2C(bboxhw_n=bboxHW_normalized, xtoc=xtoc, k=i)
            min_arg = np.argsort(dist_centers)
            print(len(xtoc))
            for k in range(i):
                print(len(np.where(xtoc==k)[0])/len(xtoc))
                print(np.var(dist[np.where(xtoc==k)[0]]))
            print(i)
            print(dist_centers[min_arg])
            print(bbox_centers[min_arg])

    def getNextBatch(self, batchSizeof3DShape=32, max_image=12, imageSize=None, gridSize=None, augmentation=True):
        if imageSize!=None:
            self._imageSize = imageSize
        if gridSize!=None:
            self._gridSize = gridSize
        inputImages, objnessImages, bbox2DImages, bbox3DImages, localXYZImages, eulerRadImages = [], [], [], [], [], []
        imageSizeList, P2List, P2InvList = [], [], []
        outputImages, instList, eulerList = [], [], []
        while len(outputImages)==0:
            for KITTITrain in self._KITTIDataListPath[self.dataStart:]:
                # try:
                objMat = scipy.io.loadmat(os.path.join(self._KITTIAnnotPath, KITTITrain + '.mat'))
                objsInfo = self._objMat2Elem(objMat)
                if len(objsInfo)>0:
                    if len(outputImages) + len(objsInfo) > batchSizeof3DShape and len(inputImages)>0:
                        break
                    if len(inputImages)>=max_image:
                        break
                    with open(os.path.join(self._KITTI_calib_path, KITTITrain+'.txt')) as fp:
                        calib = fp.readlines()
                    P2 = np.array(calib[0].split(' ')[1:])
                    projection_mat = np.identity(4)
                    for i in range(3):
                        for j in range(4):
                            projection_mat[i, j] = float(P2[4 * i + j])
                    projection_mat_inv = np.linalg.inv(projection_mat)

                    inputImage = cv2.imread(os.path.join(self._KITTIImagePath, KITTITrain + '.png'), cv2.IMREAD_COLOR)
                    imageRowOrg, imageColOrg, _ = inputImage.shape

                    is_flip, is_padding = False, False
                    if augmentation:
                        is_flip = np.random.rand() > 0.5
                        if is_flip:
                            inputImage = cv2.flip(inputImage, flipCode=1)
                        is_padding = np.random.rand() > 0.5

                    inputImage, rowPad, dRow, colPad, dCol, scale = imageRandomAugmentation(
                        inputImage=inputImage, imageRowFinal=self._imageSize[1], imageColFinal=self._imageSize[0], padding=is_padding,
                        imAug=augmentation, imAugPr=0.5,
                        randomTrans=False, randomScale=False,
                        transPr=0.5, scalePr=0.5, transRatioMaxRow=0.1, transRatioMaxCol=0.2,
                        scaleRatioMin=0.7, scaleRatioMax=1.2
                    )
                    inputImage = inputImage/255.

                    imageRow = imageRowOrg + 2 * rowPad
                    imageCol = imageColOrg + 2 * colPad

                    objOrderingImage = -1 * np.ones([self._gridSize[1], self._gridSize[0], self._predNumPerGrid])
                    objnessImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                    bbox2DImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 4])
                    bbox3DImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                    if augmentation:
                        xyzDim=1
                    else:
                        xyzDim=3
                    localXYZImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, xyzDim])
                    eulerRadImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 3])
                    outputPerImage, instPerImage, EulerPerImage = [], [], []
                    itemIndex = 0

                    # sort object by distance in z-axis
                    obj_dist_z = []
                    for objInfo in objsInfo:
                        _, _, _, _, _, _, _, _, _, local_xyz, _, _, _ = objInfo
                        dist_z = local_xyz[-1]
                        obj_dist_z.append(dist_z)
                    obj_index_list_sorted = np.argsort(-np.array(obj_dist_z))
                    for objIndex, objInfo in enumerate(objsInfo[obj_index_list_sorted]):
                        truncated, occluded, colMin, rowMin, colMax, rowMax, h, w, l, local_xyz, cadIndex, azimuth, elevation = objInfo
                        colMin, rowMin, colMax, rowMax, azimuth, elevation = float(colMin), float(rowMin),float(colMax),float(rowMax),float(azimuth),float(elevation)
                        if is_flip:
                            local_xyz = - local_xyz[0], local_xyz[1], local_xyz[2]
                            colMin, colMax = imageColOrg - colMax, imageColOrg - colMin
                            azimuth = -azimuth
                        # # border for crop and square
                        # rowMin, rowMax = float(rowMin) + heightBorderSize, float(rowMax) + heightBorderSize
                        # colMin, colMax = float(colMin) + widthBorderSize - widthCrop, float(colMax) + widthBorderSize - widthCrop
                        # augmentation
                        colMin = (float(colMin)+colPad - imageCol/2)*scale + imageCol/2 + dCol
                        colMax = (float(colMax)+colPad - imageCol/2)*scale + imageCol/2 + dCol
                        rowMin = (float(rowMin)+rowPad - imageRow/2)*scale + imageRow/2 + dRow
                        rowMax = (float(rowMax)+rowPad - imageRow/2)*scale + imageRow/2 + dRow
                        # print(occluded)
                        if augmentation :
                            if np.random.rand()<0.5:
                                cStart, cEnd, rStart, rEnd = colMin, colMax, rowMin, rowMax
                                d = np.random.rand() * 0.5
                                if np.random.rand() < 0.5: # left right
                                    if np.random.rand() < 0.5: #left
                                        cEnd = cStart + (cEnd-cStart)*d
                                    else:
                                        cStart = cEnd - (cEnd-cStart)*d
                                else: # up down
                                    if np.random.rand() < 0.5: # up
                                        rEnd = rStart + (rEnd-rStart)*d
                                    else:
                                        rStart = rEnd - (rEnd-rStart)*d
                                cStart, cEnd, rStart, rEnd = cStart/imageCol*self._imageSize[0], cEnd/imageCol*self._imageSize[0], rStart/imageRow*self._imageSize[1], rEnd/imageRow*self._imageSize[1]
                                cStart, cEnd, rStart, rEnd = int(cStart), int(cEnd), int(rStart), int(rEnd)
                                inputImage[rStart:rEnd, cStart:cEnd, :] = 0.0
                        # colMin = np.max((0.0, colMin))
                        # colMax = np.min((colMax, imageCol))
                        # rowMin = np.max((0.0, rowMin))
                        # rowMax = np.min((rowMax, imageRow))
                        # print scale, imageRow, imageCol, rowMin, rowMax, colMin, colMax
                        cadIndex = int(cadIndex)
                        h, w, l = float(h), float(w), float(l)
                        azimuth,elevation,inPlaneRot = -float(azimuth)/180.0*np.pi, -float(elevation)/180.0*np.pi, 0.0

                        rowCenterOnGrid = (rowMax + rowMin) / 2.0 * self._gridSize[1] / imageRow
                        colCenterOnGrid = (colMax + colMin) / 2.0 * self._gridSize[0] / imageCol
                        rowIndexOnGrid = int(rowCenterOnGrid)
                        colIndexOnGrid = int(colCenterOnGrid)
                        dx, dy = colCenterOnGrid - colIndexOnGrid, rowCenterOnGrid - rowIndexOnGrid
                        bboxHeight, bboxWidth = np.min((1.0, (rowMax - rowMin) / imageRow)), np.min(
                            (1.0, (colMax - colMin) / imageCol))

                        for predIndex in range(self._predNumPerGrid):
                            if (rowIndexOnGrid>=0 and rowIndexOnGrid<self._gridSize[1])\
                                and (colIndexOnGrid>=0 and colIndexOnGrid<self._gridSize[0])\
                                    and (bboxHeight >0)\
                                    and (bboxWidth > 0)\
                                    and (objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex, 0] != 1):
                                # print('add obj')
                                # objectness and bounding box
                                objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex, 0] = 1
                                bbox2DImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = bboxHeight, bboxWidth, dx, dy
                                bbox3DImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = w, h, l
                                if augmentation:
                                    localXYZImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = np.sqrt(np.sum(np.square(local_xyz)))
                                else:
                                    localXYZImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = local_xyz
                                # print(h,w,l)
                                eulerRadImage[rowIndexOnGrid, colIndexOnGrid, predIndex, :] = azimuth, elevation, inPlaneRot

                                # car instance vector
                                carInstVector = np.zeros(len(self._KITTI3DShapes))
                                carInstVector[cadIndex - 1] = 1
                                # car 3D CAD model
                                if (cadIndex - 1 < 0) or (cadIndex>len(self._KITTI3DShapes)):
                                    for i in range(1000):
                                        print('kitti ', cadIndex - 1)
                                        return
                                car3DCAD = self._KITTI3DShapes[cadIndex - 1]

                                # append items
                                outputPerImage.append(car3DCAD)
                                instPerImage.append(carInstVector)
                                EulerPerImage.append([azimuth, elevation, inPlaneRot])
                                # set item order
                                objOrderingImage[rowIndexOnGrid, colIndexOnGrid, predIndex] = itemIndex
                                itemIndex += 1

                                # rowmin = (float(rowIndexOnGrid + dy) / float(gridSize[1]) - bboxHeight / 2.0) * \
                                #          imageSize[1]
                                # rowmax = (float(rowIndexOnGrid + dy) / float(gridSize[1]) + bboxHeight / 2.0) * \
                                #          imageSize[1]
                                # colmin = (float(colIndexOnGrid + dx) / float(gridSize[0]) - bboxWidth / 2.0) * \
                                #          imageSize[0]
                                # colmax = (float(colIndexOnGrid + dx) / float(gridSize[0]) + bboxWidth / 2.0) * \
                                #          imageSize[0]
                                # p0 = (int(colmin), int(rowmin))
                                # p1 = (int(colmax), int(rowmax))
                                # color = (0, 255, 0)
                                # cv2.rectangle(img=inputImage, pt1=p0, pt2=p1, color=color, thickness=1)

                                break
                    if itemIndex > 0:

                        # cv2.imwrite('test/' + str(len(inputImages)) + '.png', inputImage)

                        inputImages.append(inputImage)
                        objnessImages.append(objnessImage)
                        bbox2DImages.append(bbox2DImage)
                        bbox3DImages.append(bbox3DImage)
                        localXYZImages.append(localXYZImage)
                        eulerRadImages.append(eulerRadImage)

                        imageSizeList.append([imageRowOrg, imageColOrg])
                        P2List.append(projection_mat)
                        P2InvList.append(projection_mat_inv)

                        for gridRow in range(self._gridSize[1]):
                            for gridCol in range(self._gridSize[0]):
                                for predIndex in range(self._predNumPerGrid):
                                    objOrder = int(objOrderingImage[gridRow, gridCol, predIndex])
                                    if objOrder >= 0:
                                        outputImages.append(outputPerImage[objOrder])
                                        instList.append(instPerImage[objOrder])
                                        eulerList.append(EulerPerImage[objOrder])
                # except:
                #     pass
                self.dataStart += 1
                if self.dataStart >= self.dataLength:  # out of dataset length
                    self.epoch += 1
                    self._dataShuffle()
                    break
        inputImages = np.array(inputImages).astype('float32')
        objnessImages = np.array(objnessImages).astype('float32')
        bbox2DImages = np.array(bbox2DImages).astype('float32')
        bbox3DImages = np.array(bbox3DImages).astype('float32')
        localXYZImages = np.array(localXYZImages).astype('float32')
        eulerRadImages = np.array(eulerRadImages).astype('float32')
        imageSizeList = np.array(imageSizeList).astype('float32')
        P2List = np.array(P2List).astype('float32')
        P2InvList = np.array(P2InvList).astype('float32')

        outputImages = np.array(outputImages).astype('float32')
        instList = np.array(instList).astype('float32')
        eulerList = np.array(eulerList).astype('float32')
        # print(inputImages.shape)
        # print(instList.shape)

        offsetX, offsetY = self._getOffset(batchSize=len(inputImages))

        # batchDict = {
        #     'offsetX': offsetX, 'offsetY': offsetY,
        #     'inputImages': inputImages,
        #     'objnessImages': objnessImages,
        #     'bbox2DImages': bbox2DImages,
        #     'bbox3DImages': bbox3DImages,
        #     'sinImages' : np.sin(eulerRadImages),
        #     'cosImages' : np.cos(eulerRadImages),
        #     'outputImages': outputImages,
        #     'instList': instList,
        # }
        # return batchDict
        # print(len(inputImages), len(outputImages))
        return offsetX, offsetY, inputImages, objnessImages,\
    bbox2DImages, bbox3DImages, localXYZImages, np.sin(eulerRadImages), np.cos(eulerRadImages),\
    imageSizeList, P2List, P2InvList,\
    outputImages, instList

# kitti = dataLoader(trainOrVal='train')
# centers = kitti.getKMeansDist(k=5)
# print(centers)

# image_size_mean = np.array([374.4770752573185, 1239.9163213474135])
# # anchor box - hw
# anchor_boxes_k2_IoU = np.array(
#     [[124.0248217, 215.79609266],
#      [38.19567016, 61.87356341]]
# )/image_size_mean
# anchor_boxes_k2_edist = np.array(
#     [[ 83.80130521, 138.40880247],
#      [ 26.3953556, 48.02598674]]
# )/image_size_mean
# anchor_boxes_k2_ldist = np.array(
#     [[ 99.4935346,  162.45685142],
#      [ 31.18130955,  56.11923566]]
# )/image_size_mean
# anchor_dist_k2_edist = np.reshape(np.array([20.30808886, 48.78342693]), (-1,1))
# anchor_dist_k2_ldist = np.reshape(np.array([16.485449,   43.05317164]), (-1,1))
# # print(anchor_boxes_k2_IoU/image_size_mean)
# anchor_boxes_k3_IoU = np.array(
#     [[151.19673255, 261.20138439],
#     [65.73346641, 116.33520631],
#      [31.12987658, 47.06349764]]
# )/image_size_mean
# anchor_boxes_k3_edist = np.array(
#     [[104.11277804, 169.50377708],
#      [37.93894797,  68.45207642],
#     [21.77237326, 38.26677287]]
# )/image_size_mean
# anchor_boxes_k3_ldist = np.array(
#     [[136.83738439, 217.08195587],
#      [50.74432051,  89.21220131],
#     [25.45746887, 46.28393874]]
# )/image_size_mean
# anchor_dist_k3_edist = np.reshape(np.array([15.62648195, 34.15850877, 56.85325256]), (-1,1))
# anchor_dist_k3_ldist = np.reshape(np.array([11.24317327, 26.52830559, 50.19864121]), (-1,1))
# anchor_boxes_k5_IoU = np.array(
#     [[176.99227226, 320.56534732],
#     [123.02909709, 198.1595076],
#     [70.27690934, 129.45642278],
#     [44.18011634, 72.90254935],
#     [26.08882807, 36.95894773]]
# )/image_size_mean
# anchor_boxes_k5_edist = np.array(
#     [[137.26860046, 217.65809066],
#      [57.29672457, 100.17431417],
#     [35.9598961, 65.20977273],
#     [24.95953267,  45.75706846],
#     [18.2335216, 30.32037959]]
# )/image_size_mean
# anchor_boxes_k5_ldist = np.array(
#     [[179.06451738, 270.93672088],
#      [96.32144439, 164.54777902],
#     [52.61116562, 90.93810029],
#     [33.83673282,  62.37542514],
#     [21.4466067, 37.51463078]]
# )/image_size_mean
# anchor_dist_k5_edist = np.reshape(np.array([11.20031385, 23.17954767, 35.2977404,  49.49552546, 66.20973349]), (-1,1))
# anchor_dist_k5_ldist = np.reshape(np.array([ 7.59584211, 15.17174581, 24.77951335, 37.59478135, 57.50771959]), (-1,1))

# kitti = dataLoader(trainOrVal='train')
# centers = kitti.getKMeansDist(k=5)

image_size_mean = np.array([374.4770752573185, 1239.9163213474135])
# anchor box - hw
anchor_boxes_k2_IoU = np.array(
    [[124.0248217, 215.79609266],
     [38.19567016, 61.87356341]]
)/image_size_mean
anchor_boxes_k2_edist = np.array(
    [[133.19241695, 218.92373942],
     [ 30.40223786, 55.29151976]]
)/image_size_mean
anchor_boxes_k2_ldist = np.array(
    [[139.28904149,  226.40121082],
     [ 37.13086686,  66.86632686]]
)/image_size_mean
anchor_boxes_k2_sdist = np.array(
    [[129.49228433,  214.76749784],
     [ 24.45300541 , 43.26812276]]
)/image_size_mean
anchor_dist_k2_edist = np.reshape(np.array([20.31439358, 48.79412723]), (-1,1))
anchor_dist_k2_ldist = np.reshape(np.array([16.485449,   43.05317164]), (-1,1))
anchor_dist_k2_sdist = np.reshape(np.array([26.29114423,   56.63224535]), (-1,1))
# print(anchor_boxes_k2_IoU/image_size_mean)
anchor_boxes_k3_IoU = np.array(
    [[151.19673255, 261.20138439],
    [65.73346641, 116.33520631],
     [31.12987658, 47.06349764]]
)/image_size_mean
anchor_boxes_k3_edist = np.array(
    [[141.19741018, 228.79474927],
     [41.53374318,  75.00528237],
    [23.82352638, 41.92541847]]
)/image_size_mean
anchor_boxes_k3_ldist = np.array(
    [[155.70978445, 245.73426512],
     [58.63193802,  103.41206887],
    [29.15007867, 52.97955665]]
)/image_size_mean
anchor_boxes_k3_sdist = np.array(
    [[133.73698917, 219.62474758],
     [32.62210371,  59.85767562],
    [20.54846429,  35.79060391]]
)/image_size_mean
anchor_dist_k3_edist = np.reshape(np.array([15.62648195, 34.15850877, 56.85325256]), (-1,1))
anchor_dist_k3_ldist = np.reshape(np.array([11.24317327, 26.52830559, 50.19864121]), (-1,1))
anchor_dist_k3_sdist = np.reshape(np.array([21.44483082, 42.47756257, 62.76250652]), (-1,1))
anchor_boxes_k5_IoU = np.array(
    [[176.99227226, 320.56534732],
    [123.02909709, 198.1595076],
    [70.27690934, 129.45642278],
    [44.18011634, 72.90254935],
    [26.08882807, 36.95894773]]
)/image_size_mean
anchor_boxes_k5_edist = np.array(
    [[155.91504255, 245.95195212],
     [63.33195305, 111.05004042],
    [38.26019577, 69.39710814],
    [26.64630765,  48.90577436],
    [19.1633743, 32.0034318]]
)/image_size_mean
anchor_boxes_k5_ldist = np.array(
    [[181.05499766, 273.30837035],
     [103.97221321, 177.59171426],
    [56.94189489, 98.57068055],
    [36.39201343,  67.05904105],
    [23.33601335, 40.86623736]]
)/image_size_mean
anchor_boxes_k5_sdist = np.array(
    [[139.59656932, 226.73198813],
     [41.68096649,  74.98155439],
    [29.27163195 , 54.33383816],
    [ 21.99570983,  38.90459293],
    [17.30815916 , 28.5362641 ]]
)/image_size_mean
anchor_dist_k5_edist = np.reshape(np.array([11.20031385, 23.17954767, 35.2977404,  49.49552546, 66.20973349]), (-1,1))
# anchor_dist_k5_edist = np.reshape(np.array([[ 9.7283461,  20.263508,   31.56858448, 44.61206565, 62.0295398 ]]), (-1,1))
anchor_dist_k5_ldist = np.reshape(np.array([ 7.59584211, 15.17174581, 24.77951335, 37.59478135, 57.50771959]), (-1,1))
anchor_dist_k5_sdist = np.reshape(np.array([ 17.49393579, 32.86485952, 45.26694211, 57.41469118, 71.51904258]), (-1,1))
anchor_boxes_k7_sdist = np.array(
    [[146.98355762, 236.50510481],
     [53.12058874 , 91.74356385],
    [37.34662169 , 68.11279995],
    [ 29.13262164 , 54.30710977],
    [23.33129122 , 41.87569647],
     [19.89880947 , 33.31455999],
     [16.55834784 , 27.34611867]]
)/image_size_mean
anchor_dist_k7_sdist = np.reshape(np.array([14.19732773, 26.11671807, 35.72234755, 44.93283893, 53.96165535, 63.00548823, 73.83235369]), (-1,1))

anchor_dist_k9_edist = np.reshape(np.array([9.03874803, 16.28369248, 22.96691315, 29.45281261, 35.71803894, 42.47122806,
 49.96879539, 58.79372826, 71.74165494]), (-1,1))
anchor_dist_k9_ldist = np.reshape(np.array([7.04594448, 12.36410579, 17.89755067, 23.65963396, 30.11189432, 36.99657865,
 45.40385668, 55.59340943, 69.37741802]), (-1,1))
anchor_dist_k9_sdist = np.reshape(np.array([12.50288999, 22.69079102, 30.33766796, 36.88770043, 43.38119544, 49.87190263,
 56.69938523, 65.00650014, 74.84226179]), (-1,1))
anchor_boxes_k9_edist = np.array(
    [[172.24350164, 263.29580958],
 [ 92.30023002, 161.15206827],
 [ 60.13132693, 103.19182539],
 [ 45.19565903,  81.00518814],
 [ 36.85228477,  66.03076853],
 [ 30.85083416,  58.93645852],
 [ 25.8387004 ,  46.04607085],
 [ 21.09253491,  37.18225934],
 [ 17.30330569,  28.55437415],]
)/image_size_mean
anchor_boxes_k9_sdist = np.array(
    [[152.85349333, 242.86351657],
 [ 62.48144755, 108.55208286],
 [ 43.97976574,  78.40155637],
 [ 35.6781033 ,  65.16825104],
 [ 30.00410208,  57.07808075],
 [ 25.91968756,  45.71759682],
 [ 21.86800761,  39.49597123],
 [ 19.0356714 ,  31.61032618],
 [ 16.33421753,  27.01973237]]
)/image_size_mean
anchor_boxes_k9_ldist = np.array(
    [[186.10117863, 279.56819826],
 [129.96070869, 211.25624039],
 [ 80.95155515, 143.90688252],
 [ 57.70495349,  98.84235557],
 [ 44.18108132,  78.60568017],
 [ 35.7255929 ,  65.43545911],
 [ 28.84426232,  53.65877296],
 [ 22.70783013,  40.30451207],
 [ 17.98034784,  29.70461492]]
)/image_size_mean




# kitti = dataLoader(trainOrVal='train')
# centers = kitti.getKMeansDist(k=5, max_iter=1000)

# simple Euclidian dist
# 0.6961064289196908
# 25.692678299076416
# 0.30389357108030923
# 31.755739638350622
# 2
# [23.71741958 52.50698276]
# [[131.49068208 216.91037581]
#  [ 28.05025191  51.12146897]]
# 0.476004736365536
# 12.256606383395075
# 0.35453089085463535
# 8.59595001763294
# 0.16946437277982865
# 20.29672681200065
# 3
# [18.07090764 36.88227327 59.08909083]
# [[138.56752947 225.45403604]
#  [ 38.48903115  70.13910159]
#  [ 22.62877485  39.46313225]]
# 0.27561468273316153
# 5.362674821660819
# 0.14745420352441319
# 4.270458020038662
# 0.28453019433029186
# 3.097432246012028
# 0.06373197743261128
# 9.278006547181441
# 0.22866894197952217
# 3.5504794804670774
# 5
# [12.97312068 25.27898039 37.43727325 51.42133346 67.55000367]
# [[150.90597274 241.05325369]
#  [ 56.8427464   98.63746622]
#  [ 36.0282949   66.21229337]
#  [ 25.49572337  45.96420134]
#  [ 18.6803172   31.1499335 ]]
# 0.14550393536254091
# 1.6109055481698027
# 0.102806993104409
# 1.899500432257148
# 0.0808664762833461
# 2.4869350580243914
# 0.18611130459009542
# 1.529878880967216
# 0.22553458243365607
# 1.8500293796822935
# 0.03587100369157902
# 5.529682443378552
# 0.22330570453437348
# 3.7574166393419515
# 7
# [11.51193363 21.71035836 30.59772299 39.09128487 48.26477206 58.20345731
#  71.51904258]
# [[157.26244172 247.34034889]
#  [ 67.66244945 118.33446607]
#  [ 44.00977419  78.3396752 ]
#  [ 33.82204298  63.36650117]
#  [ 27.10457471  48.86246024]
#  [ 21.49022614  38.23372232]
#  [ 17.30815916  28.5362641 ]]
# log-scale dist
# 14357
# 0.5467019572334053
# 0.09946218252540466
# 0.4532980427665947
# 0.015858398905155145
# 2
# [19.8422138  47.29993911]
# [[135.8949273  222.07089264]
#  [ 33.70897574  61.17486943]]
# 14357
# 0.31587378978895314
# 0.06653577869923397
# 0.4209793132269973
# 0.009368815367416483
# 0.2631468969840496
# 0.008471353625715626
# 3
# [14.05936952 30.52776351 54.25616246]
# [[147.39158038 236.99211142]
#  [ 48.77042412  85.975146  ]
#  [ 26.29430757  47.38976415]]
# 14357
# 0.12913561328968448
# 0.003561172891184529
# 0.17085742146688027
# 0.0396269155312688
# 0.24935571498223863
# 0.003039633429999276
# 0.24684822734554573
# 0.00667522663079748
# 0.2038030229156509
# 0.0024356967723427048
# 5
# [10.00638554 19.98543831 30.72284525 43.56073177 61.70020433]
# [[165.74977078 256.01827769]
#  [ 77.21409878 135.56728561]
#  [ 44.59667646  79.15042917]
#  [ 31.10967018  57.69206459]
#  [ 21.14746772  36.83955489]]
# 14357
# 0.10475726126628125
# 0.026271823992646215
# 0.16194190986974996
# 0.0013015663515234242
# 0.06408023960437417
# 0.0018198995042442096
# 0.13073761927979383
# 0.0012392320457555966
# 0.18910635926725639
# 0.0016255631754020835
# 0.15365327018179287
# 0.005240638991626044
# 0.19572334053075155
# 0.0025618242994677196
# 7
# [ 7.9347289  14.82060968 22.51365645 30.9179111  40.19590707 52.13770949
#  67.50848041]
# [[180.52451843 272.4515248 ]
#  [105.71780052 179.81213836]
#  [ 62.85534169 109.17059693]
#  [ 43.44428907  77.35104273]
#  [ 33.07188473  61.93281268]
#  [ 24.85351193  44.51941171]
#  [ 18.69346263  31.16884911]]
#
# Process finished with exit code 0
