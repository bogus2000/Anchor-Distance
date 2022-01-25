import scipy.io
import os
import numpy as np
import cv2
from src.dataset_loader.datasetUtils import *



# average image size of pascal3D : (508.54, 404.47)
class Pascal3DMultiObject(object):
    def __init__(self,
                 imageSize=(640,480),
                 gridSize=(20,15),
                 predNumPerGrid=5,
                 Pascal3DDataPath=None,
                 trainOrVal='train',
                 ):
        self.dataStart = 0
        self.dataLength = 0
        self.epoch = 0

        self._imageSize = imageSize
        self._gridSize = gridSize
        self._predNumPerGrid = predNumPerGrid
        self._Pascal3DDataPath = Pascal3DDataPath
        self._trainOrVal = trainOrVal
        self._isTrain = True
        if self._trainOrVal == 'train':
            self._isTrain = True
        elif self._trainOrVal == 'val':
            self._isTrain = False
        else:
            print('choose \'train\' or \'val\'')
            return
        self._dataPathList = []
        self._CAD3DShapes = None

        print('set pascal3d dataset...')

        self._getTrainList()
        self._loadDataPath()
        self._load3DShapes()
        self._dataPathShuffle()

    def _getTrainList(self):
        print('set train or val list...')
        self._Pascal3DTrainList = []
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/'))
        for datasetName in datasetList:
            if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/', datasetName)):
                txtFileList = os.listdir(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName))
                for txtFileName in txtFileList:
                    className = txtFileName.split('.')[0].split('_')[0]
                    trainval = txtFileName.split('.')[0].split('_')[-1]
                    if className=='car' and trainval==self._trainOrVal:
                        with open(os.path.join(self._Pascal3DDataPath, 'Image_sets/',datasetName,txtFileName)) as txtFilePointer:
                            dataPointList = txtFilePointer.readlines()
                            for i, dataPoint in enumerate(dataPointList):
                                if datasetName == 'pascal':
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    isTrue = int(dataPoint.split('\n')[0].split(' ')[-1])
                                    if int(isTrue)==1:
                                        self._Pascal3DTrainList.append(dp)
                                else:
                                    dp = dataPoint.split('\n')[0].split(' ')[0]
                                    self._Pascal3DTrainList.append(dp)
        print('done!')

    def _loadDataPath(self):
        print('load datapoint path...')
        datasetList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data'))
        for datasetName in datasetList:
            if datasetName == 'imagenet' or datasetName == 'pascal':
                dataPointList = os.listdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName))
                for dataPointName in dataPointList:
                    if dataPointName in self._Pascal3DTrainList:
                        if os.path.isdir(os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)):
                            dataPointPath = os.path.join(self._Pascal3DDataPath, 'training_data', datasetName, dataPointName)
                            self._dataPathList.append(dataPointPath)
        self._dataPathList = np.array(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        print('done!')

    def _dataPathShuffle(self):
        print('')
        print('data path shuffle...')
        self.dataStart = 0
        np.random.shuffle(self._dataPathList)
        self.dataLength = len(self._dataPathList)
        print('done! : ' + str(self.dataLength))

    def _load3DShapes(self):
        print('load 3d shapes for pascal3d...')
        self._CAD3DShapes = []
        CADModelList = os.listdir(os.path.join(self._Pascal3DDataPath, 'CAD', 'car'))
        CADModelList.sort()
        for CADModel in CADModelList:
            if CADModel.split(".")[-1] == 'npy':
                shape = np.load(os.path.join(self._Pascal3DDataPath, 'CAD', 'car', CADModel)).reshape(64, 64, 64, 1)
                self._CAD3DShapes.append(shape)
        self._CAD3DShapes = np.array(self._CAD3DShapes)
        self._CAD3DShapes = np.where(self._CAD3DShapes>0, 1.0, 0.0)
        print('done!')

    def _getOffset(self, batchSize):
        offsetX = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[0])]*self._gridSize[1]*self._predNumPerGrid),
            (self._predNumPerGrid, self._gridSize[1], self._gridSize[0])), (1,2,0))
        offsetX = np.tile(np.reshape(offsetX, (1,self._gridSize[1],self._gridSize[0],self._predNumPerGrid)),[batchSize,1,1,1])
        offsetY = np.transpose(np.reshape(
            np.array([np.arange(self._gridSize[1])]*self._gridSize[0]*self._predNumPerGrid),
            (self._predNumPerGrid, self._gridSize[0], self._gridSize[1])), (2,1,0))
        offsetY = np.tile(np.reshape(offsetY, (1,self._gridSize[1],self._gridSize[0],self._predNumPerGrid)),[batchSize,1,1,1])
        return offsetX.astype('float'), offsetY.astype('float')

    def getNextBatch(self, batchSizeof3DShape=32, imageSize=None, gridSize=None):
        if imageSize!=None:
            self._imageSize = imageSize
        if gridSize!=None:
            self._gridSize = gridSize
        inputImages, bboxImages, objnessImages = [], [], []
        outputImages, instList, EulerRadList = [], [], []
        while len(outputImages)==0:
            for dataPath in self._dataPathList[self.dataStart:]:
                objFolderList = os.listdir(dataPath)
                np.random.shuffle(objFolderList)
                # objFolderList.sort()
                objSelectedList = []
                for objFolder in objFolderList:
                    if os.path.isdir(os.path.join(dataPath, objFolder)):
                        objInfoTXT = os.path.join(dataPath, objFolder, 'objInfo.txt')
                        with open(objInfoTXT) as objInfoPointer:
                            objInfo = objInfoPointer.readline()
                        className = objInfo.split(' ')[0]
                        if className == 'car':
                            objSelectedList.append(objInfo)
                if len(objSelectedList) > 0:
                    if len(outputImages) + len(objSelectedList) > batchSizeof3DShape and len(inputImages)>0:
                        break
                    try:
                        image2DPath = os.path.join(self._Pascal3DDataPath, objSelectedList[0].split(' ')[1])
                        inputImage = cv2.imread(image2DPath, cv2.IMREAD_COLOR)

                        # imageRow, imageCol, channel = inputImage.shape
                        # # make this a square image
                        # imgRowColMax = np.max((imageRow, imageCol))
                        # heightBorderSize = (imgRowColMax - imageRow) / 2
                        # widthBorderSize = (imgRowColMax - imageCol) / 2
                        # inputImage = cv2.copyMakeBorder(
                        #     inputImage, top=heightBorderSize, bottom=heightBorderSize,
                        #     left=widthBorderSize, right=widthBorderSize, borderType=cv2.BORDER_CONSTANT,
                        #     value=[0, 0, 0])

                        imageRowOrg, imageColOrg, _ = inputImage.shape
                        inputImage, rowPad, dRow, colPad, dCol, scale = imageRandomAugmentation(
                            inputImage=inputImage, imageRowFinal=self._imageSize[1], imageColFinal=self._imageSize[0],
                            imAug=self._isTrain, imAugPr=0.5,
                            randomTrans=self._isTrain, randomScale=self._isTrain,
                            transPr=0.5, scalePr=0.5, transRatioMax=0.2,
                            scaleRatioMin=0.8, scaleRatioMax=1.2
                        )
                        imageRow = imageRowOrg + 2*rowPad
                        imageCol = imageColOrg + 2*colPad

                        objOrderingImage = -1 * np.ones([self._gridSize[1], self._gridSize[0], self._predNumPerGrid])
                        bboxImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 4])
                        objnessImage = np.zeros([self._gridSize[1], self._gridSize[0], self._predNumPerGrid, 1])
                        outputPerImage, instPerImage, EulerPerImage = [], [], []
                        itemIndex = 0
                        for objIndex, objSelected in enumerate(objSelectedList):
                            className,imagePath,CADModelPath,colMin,rowMin,colMax,rowMax,azimuth,elevation,inPlaneRot=objSelected.split(' ')
                            # #border for square
                            # rowMin, rowMax = float(rowMin) + heightBorderSize, float(rowMax) + heightBorderSize
                            # colMin, colMax = float(colMin) + widthBorderSize, float(colMax) + widthBorderSize
                            # augmentation
                            colMin = (float(colMin) + colPad - imageCol/2) * scale + imageCol/2 + dCol
                            colMax = (float(colMax) + colPad - imageCol/2) * scale + imageCol/2 + dCol
                            rowMin = (float(rowMin) + rowPad - imageRow/2) * scale + imageRow/2 + dRow
                            rowMax = (float(rowMax) + rowPad - imageRow/2) * scale + imageRow/2 + dRow
                            # colMin = np.max((0.0, colMin))
                            # colMax = np.min((colMax, imageCol))
                            # rowMin = np.max((0.0, rowMin))
                            # rowMax = np.min((rowMax, imageRow))
                            azimuth,elevation,inPlaneRot = float(azimuth)/180.0*np.pi,float(elevation)/180.0*np.pi,float(inPlaneRot)/180.0*np.pi
                            cadIndex = int(CADModelPath.split('/')[-1])

                            rowCenterOnGrid = (rowMax+rowMin)/2.0*self._gridSize[1]/imageRow
                            colCenterOnGrid = (colMax+colMin)/2.0*self._gridSize[0]/imageCol
                            rowIndexOnGrid = int(rowCenterOnGrid)
                            colIndexOnGrid = int(colCenterOnGrid)
                            dx,dy = colCenterOnGrid - colIndexOnGrid, rowCenterOnGrid - rowIndexOnGrid
                            bboxHeight,bboxWidth = np.min((1.0, (rowMax-rowMin)/imageRow)),np.min((1.0, (colMax-colMin)/imageCol))
                            for predIndex in range(self._predNumPerGrid):
                                if (rowIndexOnGrid>=0 and rowIndexOnGrid<self._gridSize[1]) \
                                    and (colIndexOnGrid>=0 and colIndexOnGrid<self._gridSize[0]) \
                                    and bboxHeight > 0 and bboxWidth > 0 \
                                    and objnessImage[rowIndexOnGrid, colIndexOnGrid, predIndex] != 1:
                                    # objectness and bounding box
                                    objnessImage[rowIndexOnGrid,colIndexOnGrid,predIndex]=1
                                    bboxImage[rowIndexOnGrid,colIndexOnGrid,predIndex,0:4] = bboxHeight,bboxWidth,dx,dy
                                    # car instance vector
                                    carInstVector = np.zeros(len(self._CAD3DShapes))
                                    carInstVector[cadIndex-1] = 1
                                    # car 3d shape
                                    if(cadIndex-1<0) or (cadIndex>len(self._CAD3DShapes)):
                                        for i in range(1000):
                                            print('pascal ', cadIndex-1)
                                            return
                                    car3DCAD = self._CAD3DShapes[cadIndex-1]
                                    # Euler angle in rad
                                    EulerRad = np.array([azimuth,elevation,inPlaneRot])

                                    # append items
                                    outputPerImage.append(car3DCAD)
                                    instPerImage.append(carInstVector)
                                    EulerPerImage.append(EulerRad)
                                    # set item order
                                    objOrderingImage[rowIndexOnGrid, colIndexOnGrid, predIndex] = itemIndex
                                    itemIndex += 1
                                    break
                        if itemIndex > 0:
                            inputImages.append(inputImage)
                            bboxImages.append(bboxImage)
                            objnessImages.append(objnessImage)

                            for gridRow in range(self._gridSize[1]):
                                for gridCol in range(self._gridSize[0]):
                                    for predIndex in range(self._predNumPerGrid):
                                        objOrder = int(objOrderingImage[gridRow, gridCol, predIndex])
                                        if objOrder>=0:
                                            outputImages.append(outputPerImage[objOrder])
                                            instList.append(instPerImage[objOrder])
                                            EulerRadList.append(EulerPerImage[objOrder])
                    except:
                        pass
                self.dataStart += 1
                if self.dataStart >= self.dataLength:
                    self.epoch += 1
                    self._dataPathShuffle()
                    break
        inputImages = np.array(inputImages).astype('float')
        bboxImages = np.array(bboxImages).astype('float')
        objnessImages = np.array(objnessImages).astype('float')
        outputImages = np.array(outputImages).astype('float')
        instList = np.array(instList).astype('float')
        EulerRadList = np.array(EulerRadList).astype('float')
        offsetX,offsetY = self._getOffset(batchSize=len(inputImages))

        # print inputImages.shape
        # print instList.shape
        return offsetX, offsetY, inputImages, objnessImages,\
    bboxImages, np.sin(EulerRadList), np.cos(EulerRadList), \
    outputImages, instList
























