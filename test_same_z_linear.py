import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

import src.dataset_loader.KITTI_dataset_p2 as kitti
import src.net_core.darknet as Darknet
import src.module.nolbo_test as nolbo
import tensorflow as tf
import cv2

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

imageSizeAndBatchListKITTI = [
    # # [384,128,64],
    # # [480,160,64],
    # [576,192,50,16],
    # [672,224,50,14],
    # [768,256,50,14],
    # [864,288,48,14],
    # [960,288,48,12],
    # [960,320,48,10],
    # [960,352,38,10],
    # [1056,352,42,10],
    # [1088,352,42,10],
    # [1088,384,40,10],
    # [1152,384,40,9],
    [1216,384,128,10],
    # [1216,448,40,9],
    # [1248,384,42,8],
    # [1248,448,42,8],
    # # #############################
    # [160,128,64],
    # [288,288,56],
    # [320,256,48],
    # [320,320,48],
    # [416,416,40],
    # [480,384,40],
    # [448,448,40],
    # [640,512,32],
    # [800,640,24],
]

# imageSizeAndBatchListKITTI = [
#     # [384,128,14],
#     # [480,160,12],
#     # [576,192,12],
#     # [672,224,10],
#     # [768,256,10],
#     # [864,288,40],
#     # [960,288,38],
#     # [960,320,36],
#     # [1056,352,36],
#     [1152,384,32],
#     [1216,384,32],
# ]

def train(
        training_epoch = 1000,
        learning_rate = 1e-4,
        config = None, predictor_num=5,
        save_path = None, load_path = None,
        load_encoder_backbone_path = None, load_encoder_backbone_name = None,
        load_decoder_path = None, load_decoder_name = None,
):
    data_loader = kitti.dataLoader(predNumPerGrid=predictor_num, trainOrVal='val', is_train=False)
    model = nolbo.nolbo_test(nolbo_structure=config,
                             backbone_style=Darknet.Darknet19)
    # model = nolbo.nolbo(nolbo_structure=config,
    #                     backbone_style=Darknet.Darknet19,
    #                     learning_rate=learning_rate,
    #                     IoU2D_loss=False, IoU3D_loss=False, exp=False,
    #                     sqr_scale=True)

    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        print('done!')

    epoch, epoch_curr = 0, 0

    print('start training...')
    KITTIImagePath = '/home/yonsei/dataset/KITTI/data_object_image_2/training/image_2/'
    KITTI_calib_path = '/home/yonsei/dataset/KITTI/data_object_calib/training/calib/'
    KITTIDataListPath = '/home/yonsei/dataset/KITTI/trainvalsplit_3DOP_MONO3D/'
    valPath = os.path.join(KITTIDataListPath, 'val.txt')
    KITTIDataListPath = np.loadtxt(valPath, dtype='str')
    for index, KITTIData in enumerate(KITTIDataListPath):
        with open(os.path.join(KITTI_calib_path, KITTIData + '.txt')) as fp:
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
        inputImage = cv2.imread(os.path.join(KITTIImagePath, KITTIData + '.png'), cv2.IMREAD_COLOR)
        image_size = inputImage.shape[:2]
        inputImage_net = cv2.resize(inputImage, (1216,384))
        inputImage_net = inputImage_net/255.0

        gridSize = int(1216 / 32), int(384 / 32)
        anchor_path = '/home/yonsei/pyws/NOLBO_grid_anchor_no3Ddec/trash/NOLBO_2D_3D_IoU_exp_viewing/src/dataset_loader/anchor_same_z_'+str(predictor_num)+'/'
        anchor_global_path = '/home/yonsei/pyws/NOLBO_grid_anchor_no3Ddec/trash/NOLBO_2D_3D_IoU_exp_viewing/src/dataset_loader/anchor_global_'+str(predictor_num)+'/'
        anchor_z_global = np.load(anchor_global_path + 'global_anchor_xyz_s.npy')
        anchor_bbox3D_global = np.load(anchor_global_path + 'global_anchor_bbox3D_s.npy')
        anchor_xyz = np.load(anchor_path + 'anchor_xyz_{:d}{:d}.npy'.format(gridSize[0], gridSize[1]))
        anchor_xyz[:,:,:,-1] = anchor_z_global[:,-1]
        anchor_ry = np.load(anchor_path + 'anchor_ry_{:d}{:d}.npy'.format(gridSize[0], gridSize[1]))
        anchor_alpha = np.load(anchor_path + 'anchor_alpha_{:d}{:d}.npy'.format(gridSize[0], gridSize[1]))
        anchor_bbox3D = np.load(anchor_path + 'anchor_bbox3D_{:d}{:d}.npy'.format(gridSize[0], gridSize[1]))
        anchor_bbox3D[:,:,:,:] = anchor_bbox3D_global[:,:]

        inputImages = np.array([inputImage_net])
        P2List = np.array([projection_mat])
        P2InvList = np.array([projection_mat_inv])

        objsPose, objsBbox3DSize, bbox3D8Points, objsBbox2D, rys, alphas, objness = \
            model.getPred(
                input_image=inputImages,
                P2_gt=P2List, P2_inv_gt=P2InvList,
                image_size=image_size, obj_thresh=0.01, IOU_thresh=0.1, is_exp=False,
                anchor_xyz=anchor_xyz, anchor_bbox3D=anchor_bbox3D, anchor_theta=anchor_ry
            )

        def draw2Dbbox(image, bbox2d, color=(0, 255, 0), thickness=2):
            p0 = (bbox2d[0], bbox2d[1])
            p1 = (bbox2d[2], bbox2d[3])
            cv2.rectangle(image, p0, p1, color=color, thickness=thickness)
        # print(imageSizeList[0,0,0,0])
        # img = cv2.resize(inputImages[0]*255, (int(imageSizeList[0,0,0,0,1]), int(imageSizeList[0,0,0,0,0])))

        fp = open(save_path + 'data/{:06d}.txt'.format(index), "w")
        print(index, KITTIData)
        if len(objsPose) > 0:
            for pose, b3DDim, box2D, alpha, ry, objn in zip(objsPose, objsBbox3DSize, objsBbox2D, alphas, rys, objness):
                # print(file_name_list[0])
                txtline = [
                    alpha[0],
                    box2D[0], box2D[1], box2D[2], box2D[3],
                    b3DDim[0], b3DDim[1], b3DDim[2],
                    pose[0,-1], pose[1,-1], pose[2,-1],
                    ry[0], objn[0]
                ]
                # print(txtline)
                fp.write("Car {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                    0., int(0),
                    alpha[0],
                    box2D[0], box2D[1], box2D[2], box2D[3],
                    b3DDim[1], b3DDim[2], b3DDim[0], #l,h,w to h,w,l
                    pose[0,-1], pose[1,-1] + b3DDim[1]/2., pose[2,-1], #y -> y + h/2
                    ry[0], objn[0]))
                draw2Dbbox(inputImage, box2D)
        fp.close()
        # print(save_path + 'image/' + file_name_list[0] + '.png')
        cv2.imwrite(save_path + 'image/' + KITTIData + '.png', inputImage)



predictor_num = 12
# latent_dim = 16
# inst_num = 10
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
        'output_dim' : predictor_num*(1+2+3+(2*1+2)+(2+1)),  # objness + b2Dxy + lhw + (2*z + logvarxy) + (sincos+var)
        'filter_num_list':[1024,1024,1024,1024,512,512,512],
        'filter_size_list':[3,3,3,3,1,1,1],
        'activation':'elu',
    },
    # 'decoder':{
    #     'name':'decoder',
    #     'input_dim' : latent_dim,
    #     'output_shape':[64,64,64,1],
    #     'filter_num_list':[512,256,128,64,1],
    #     'filter_size_list':[4,4,4,4,4],
    #     'strides_list':[1,2,2,2,2],
    #     'activation':'elu',
    #     'final_activation':'sigmoid'
    # },
    # 'prior' : {
    #     'name' : 'priornet',
    #     'input_dim' : inst_num,  # class num (one-hot vector)
    #     'unit_num_list' : [32, 16],
    #     'core_activation' : 'elu',
    #     'const_log_var' : 0.0,
    # }
}

if __name__ == '__main__':
    sys.exit(train(
        training_epoch=1000, learning_rate=1e-4,
        config=config, predictor_num=predictor_num,
        save_path='./data/same_z_linear/',
        load_path='./weights_dist_exp/same_z_linear/',
        # load_encoder_backbone_path='./weights/imagenet_and_place365_and_Object/',
        # load_encoder_backbone_name='imagenet_backbone',
        # load_decoder_path='./weights/imagenet_and_place365_and_Object/',
        # load_decoder_name='decoder',
    ))



