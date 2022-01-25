import numpy as np
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

import src.dataset_loader.KITTI_dataset_p2_mat as kitti
import src.net_core.darknet as Darknet
import src.module.nolbo as nolbo
import tensorflow as tf
import cv2

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

imageSizeAndBatchListKITTI = [
    # # [384,128,64],
    # # [480,160,64],
    # [576,192,64,26],
    # [672,224,64,24],
    # [768,256,64,22],
    # [864,288,52,14],
    # [960,288,50,14],
    # [960,320,48,14],
    # [960,352,46,12],
    # [1056,352,46,10],
    # [1088,352,46,10],
    # [1088,384,44,10],
    # [1152,384,42,8],
    [1216,448,42,6],
    # [1216,448,64,8],
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

def train(
        training_epoch = 1000,
        learning_rate = 1e-4, solver='adam',
        config = None, predictor_num=5,
        save_path = None, load_path = None,
        load_encoder_backbone_path = None, load_encoder_backbone_name = None,
        load_decoder_path = None, load_decoder_name = None,
        data_path='',
):
    # data_path = '/home/yonsei/pyws/NOLBO_grid_anchor_no3Ddec/NOLBO_2D_3D_IoU_exp_viewing/data/kitti_split1_org'
    data_loader = kitti.dataLoader(predNumPerGrid=predictor_num,
                                   KITTIImagePath=os.path.join(data_path, 'training/image_2/'),
                                   KITTIAnnotPath=os.path.join(data_path, 'training/3dv_2/'),
                                   KITTI_calib_path=os.path.join(data_path, 'training/calib/'),
                                   KITTI3DShapePath='/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/CAD/car/',
                                   KITTI_anchor_path=data_path
                                   )
    model = nolbo.nolbo(nolbo_structure=config, solver=solver,
                        learning_rate=learning_rate,
                        IoU2D_loss=False, IoU3D_loss=False)

    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        # model.loadEncoder(load_path=load_path)
        # model.loadEncoderBackbone(load_path=load_path)
        # model.loadEncoderHead(load_path=load_path)
        # model.loadEncoderHeadTop(load_path=load_path)
        # model.loadDecoder(load_path=load_path)
        # model.loadPriornet(load_path=load_path)
        print('done!')

    if load_encoder_backbone_path != None:
        print('load encoder backbone weights...')
        model.loadEncoderBackbone(
            load_path=load_encoder_backbone_path,
            file_name=load_encoder_backbone_name
        )
        print('done!')

    if load_decoder_path != None:
        print('load decoder weights...')
        model.loadDecoder(
            load_path=load_decoder_path,
            file_name=load_decoder_name
        )
        print('done!')

    loss = np.zeros(14)
    epoch, iteration, run_time = 0., 0., 0.

    item_num = 0
    print('start training...')
    while epoch < training_epoch:
        start_time = time.time()

        periodOfImageSize = 3
        if int(iteration) % (periodOfImageSize * len(imageSizeAndBatchListKITTI)) == 0:
            np.random.shuffle(imageSizeAndBatchListKITTI)
        image_col, image_row, _, batch_size = imageSizeAndBatchListKITTI[
            int(iteration) % int((periodOfImageSize * len(imageSizeAndBatchListKITTI)) / periodOfImageSize)]
        image_size = image_col, image_row

        batch_data = data_loader.getNextBatch(batchSize=batch_size,
                                              imageSize=image_size,
                                              gridSize=(int(image_size[0] / 32), int(image_size[1] / 32)),
                                              augmentation=True, dist_type='s')
        # images = batch_data[2]
        # for image in images:
        #     cv2.imwrite(os.path.join('./test/image/', '{:03d}.png'.format(item_num)), image)
        #     item_num += 1
        epoch_curr = data_loader.epoch
        data_start = data_loader.dataStart
        data_length = data_loader.dataLength

        if epoch_curr != epoch:
            print('')
            iteration = 0
            loss = loss * 0.
            run_time = 0.
            if save_path != None:
                print('save model...')
                model.saveModel(save_path=save_path)
        epoch = epoch_curr
        # try:
        loss_temp = model.fit(inputs=batch_data)

        end_time = time.time()

        loss = (loss * iteration + np.array(loss_temp)) / (iteration + 1.0)
        run_time = (run_time * iteration + (end_time - start_time)) / (iteration + 1.0)
        sys.stdout.write(
            "Ep:{:03d} it:{:04d} rt:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:04d}/{:04d} b:{:02d} ".format(data_start, data_length, len(batch_data[0])))
        sys.stdout.write(
            "obj:{:.2f}, noobj:{:.2f}, b2D:{:.3f}, b3DI:{:.3f}, ".format(loss[0], loss[1], loss[2], loss[3]))
        sys.stdout.write(
            "b3Dd:{:.3f}, loc:{:.3f} ".format(loss[4], loss[5]))
        sys.stdout.write(
            "sc:{:.2f}, sc1:{:.2f} ".format(loss[6], loss[7]))
        sys.stdout.write(
            "ca:{:.2f}, sh:{:.2f} ".format(loss[8], loss[9]))
        sys.stdout.write(
            "op:{:.2f}, np:{:.2f} ".format(loss[10], loss[11]))
        sys.stdout.write(
            "pr:{:.3f}, rc:{:.3f}    \r".format(loss[12], loss[13]))
        sys.stdout.flush()

        if np.sum(loss) != np.sum(loss):
            print('')
            print('NaN')
            return
        iteration += 1.0

        # except:
        # #     pass
        #     print('save model...')
        #     model.saveModel(save_path=save_path)
        #     print(image_col, image_row, len(batch_data[0]), len(batch_data[-4]))
        #     return

predictor_num = 12
latent_dim = 64
category_num = kitti.category_num
car_instance_num = kitti.car_instance_num
config = {
    'category_num':category_num,
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num': predictor_num,
        'bbox2DXY_dim':2, 'bbox3D_dim':3, 'localXYZ_dim':1, 'orientation_dim':1,
        'latent_dim': latent_dim,
        'activation' : 'lrelu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : predictor_num*(1+2+3+1+2+ 2*latent_dim),  # objness + b2Dxy + lhw + z + sincos
        'activation':'relu',
    },
    'decoder':{
        'name':'decoder',
        'input_dim' : latent_dim,
        'output_shape':[64,64,64,1],
        'filter_num_list':[512,256,128,64,1],
        'filter_size_list':[4,4,4,4,4],
        'strides_list':[1,2,2,2,2],
        'activation':'relu',
        'final_activation':'sigmoid'
    },
    'prior' : {
        'name' : 'priornet',
        'input_dim' : car_instance_num,  # class num (one-hot vector)
        'unit_num_list' : [latent_dim//2, latent_dim],
        'core_activation' : 'relu',
        'const_log_var' : 0.0,
    }
}
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':
    sys.exit(train(
        training_epoch=20, learning_rate=1e-4, solver='adam',
        config=config, predictor_num=predictor_num,
        save_path='./weights/kitti_split1_org/',
        load_path='./weights/kitti_split1_org/',
        # load_encoder_backbone_path='./weights/yolov2/',
        # load_encoder_backbone_name='nolbo_backbone',
        # load_decoder_path='./weights/AE3D/',
        # load_decoder_name='decoder3D',
        data_path='./data/kitti_split1',
    ))



