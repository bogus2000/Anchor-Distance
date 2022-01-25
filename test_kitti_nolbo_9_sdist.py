import numpy as np
import time, sys, os, scipy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 0 = all messages are logged(default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages  arenot printed
# 3 = INFO, WARNING, and ERROR messages  arenot printed

import src.dataset_loader.KITTI_dataset_p2 as kitti
import src.net_core.darknet as Darknet
import src.module.nolbo_test as nolbo
import tensorflow as tf

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

def xyzEval(gt, pred):
    gt_z = np.reshape(gt[:, -1], [-1, 1])
    pred_z = np.reshape(pred[:, -1], [-1, 1])
    thresh = np.max(np.concatenate([gt_z / (pred_z + 1e-9), pred_z / (gt_z + 1e-9)], axis=-1), axis=-1)
    acc125 = np.mean(np.where(thresh < 1.25, 1., 0.))
    acc1252 = np.mean(np.where(thresh < 1.25 ** 2, 1., 0.))
    acc1253 = np.mean(np.where(thresh < 1.25 ** 3, 1., 0.))
    acc = np.array([acc125, acc1252, acc1253])

    ABS_rel = np.mean(np.abs(gt[:, -1] - pred[:, -1]) / (gt[:, -1] + 1e-9))
    SQR_rel = np.mean(np.square(gt[:, -1] - pred[:, -1]) / (gt[:, -1] + 1e-9))

    RMSE_log = np.sqrt(np.mean(np.square(np.log(gt[:, -1] + 1e-9) - np.log(pred[:, -1] + 1e-9))))
    RMSE_lin = np.sqrt(np.mean(np.square(gt - pred), axis=0))

    error = np.abs(pred - gt)
    dist = np.sqrt(np.sum(np.square(gt), axis=-1))
    idx = np.argsort(dist)
    dist_sorted = np.reshape(dist[idx], [-1, 1])
    error_sorted = np.reshape(error[idx], [-1, 3])
    error_dist = np.concatenate([dist_sorted, error_sorted], axis=-1)

    return acc, ABS_rel, SQR_rel, RMSE_log, RMSE_lin, error_dist

def geodesicDist(R1,R2):
    R1tR2 = np.matmul(R1.transpose(), R2)
    logR = scipy.linalg.logm(R1tR2)
    frobNorm = np.linalg.norm(logR, 'fro')
    dR = frobNorm/1.414213
    return dR

def SCEval(sc_gt_list, sc_pred_list):
    err = []
    i = 0
    for sc_gt, sc_pr in zip(sc_gt_list, sc_pred_list):
        print('R_eval', i+1, len(sc_gt_list))
        i += 1
        s_gt, c_gt = sc_gt
        R_gt = np.array([[c_gt, 0., s_gt],
                         [0., 1., 0.],
                         [-s_gt, 0., c_gt]])
        s_pr, c_pr = sc_pr
        R_pr = np.array([[c_pr, 0., s_pr],
                         [0., 1., 0.],
                         [-s_pr, 0., c_pr]])
        err.append(geodesicDist(R_gt, R_pr))
    acc_4pi = np.sum(np.where(np.array(err) < np.pi / 4.0, 1.0, 0.0)) / float(len(err))
    acc_6pi = np.sum(np.where(np.array(err) < np.pi / 6.0, 1.0, 0.0)) / float(len(err))
    med_err = np.median(err)*180.0/3.14159265358979
    return acc_4pi, acc_6pi, med_err

def train(
        config = None, predictor_num=5,
        save_path = None, load_path = None,
        data_path=None, batch_size=None, image_size=None,
        is_bayesian=False,
):
    data_loader = kitti.dataLoader(predNumPerGrid=predictor_num,
                                   KITTIImagePath=os.path.join(data_path, 'validation/image_2/'),
                                   KITTIAnnotPath=os.path.join(data_path, 'validation/label_2/'),
                                   KITTI_calib_path=os.path.join(data_path, 'validation/calib/'),
                                   KITTI_anchor_path=data_path
                                   )
    model = nolbo.nolboXYZEval(nolbo_structure=config,
                               backbone_style=Darknet.Darknet19,
                               is_bayesian=is_bayesian)

    if load_path != None:
        print('load weights...')
        model.loadModel(load_path=load_path)
        print('done!')

    localXYZ_gt_list, localXYZ_estimated_list = [], []
    sc_gt_list, sc_pr_list = [], []

    print('start evaluation...')
    epoch = 0
    iteration, run_time = 0.0, 0.0
    while epoch < 1:
        start_time = time.time()
        batch_data = data_loader.getNextBatch(batchSize=batch_size,
                                              imageSize=image_size,
                                              gridSize=(int(image_size[0] / 32), int(image_size[1] / 32)),
                                              augmentation=False, dist_type='s')
        _, _, inputImages, objnessImages, \
        bbox2DImages, bbox2DXYImages, bbox3DDimImages, localXYZImages, alphaImages, bbox3D8PointsImages, imageSizeList, P2List, P2InvList, anchor_z_global, anchor_bbox3D = batch_data
        epoch_curr = data_loader.epoch
        data_start = data_loader.dataStart
        data_length = data_loader.dataLength
        if epoch_curr != epoch:
            break
        epoch = epoch_curr
        # try:
        localXYZ_gt, localXYZ_estimated, sc_gt, sc_pr = model.getEval(
            inputImages=inputImages, anchor_bbox3D=anchor_bbox3D, anchor_z=anchor_z_global,
            objnessImages_gt=objnessImages, bbox3DImages_gt=bbox3DDimImages, localXYZImages_gt=localXYZImages,
            sin_gt=np.sin(alphaImages), cos_gt=np.cos(alphaImages), imageSizeImages=imageSizeList,
            P2Images=P2List, P2InvImages=P2InvList
        )
        localXYZ_gt_list.append(localXYZ_gt)
        localXYZ_estimated_list.append(localXYZ_estimated)
        sc_gt_list.append(sc_gt)
        sc_pr_list.append(sc_pr)
        end_time = time.time()

        run_time = (run_time * iteration + (end_time - start_time)) / (iteration + 1.0)
        sys.stdout.write(
            "Epoch:{:03d} iteration:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("current/total:{:04d}/{:04d} batch_size:{}   \r".format(data_start, data_length, len(batch_data[0])))
        sys.stdout.flush()
        iteration += 1.0
    print('xyz concat')
    localXYZ_gt_list = np.concatenate(localXYZ_gt_list, axis=0)
    localXYZ_estimated_list = np.concatenate(localXYZ_estimated_list, axis=0)
    print('r concat')
    sc_gt_list = np.concatenate(sc_gt_list)
    sc_pr_list = np.concatenate(sc_pr_list)
    print('save')
    np.save(os.path.join(save_path, 'gt.npy'), localXYZ_gt_list)
    np.save(os.path.join(save_path, 'estimated.npy'), localXYZ_estimated_list)
    np.save(os.path.join(save_path, 'sc_gt.npy'), sc_gt_list)
    np.save(os.path.join(save_path, 'sc_pr.npy'), sc_pr_list)
    print('eval')
    acc, ABS_rel, SQR_rel, RMSE_log, RMSE_lin, error_dist = xyzEval(gt=localXYZ_gt_list, pred=localXYZ_estimated_list)
    acc_4pi, acc_6pi, med_err = SCEval(sc_gt_list=sc_gt_list, sc_pred_list=sc_pr_list)
    txt = np.array([[acc[0], acc[1], acc[2], acc_4pi],
                    [ABS_rel, SQR_rel, RMSE_log, acc_6pi],
                    [RMSE_lin[0], RMSE_lin[1], RMSE_lin[2], med_err]])
    txt = np.concatenate([txt, error_dist], axis=0)
    np.savetxt(os.path.join(save_path, 'eval.txt'), txt)


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
config_bayesian = {
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
        'output_dim' : predictor_num*(1+2+3+2+3),  # objness + b2Dxy + lhw + z+zlogvar + sincos+radlogvar
        'filter_num_list':[1024,1024,1024],
        'filter_size_list':[3,3,3],
        'activation':'elu',
    }
}
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':
    # train(
    #     config=config, predictor_num=predictor_num,
    #     save_path='./eval/split2_added_full/',
    #     load_path='./weights/kitti_split2_added_full/',
    #     data_path='./data/kitti_split2_added_full',
    #     batch_size=32, image_size=(1216, 448)
    # )
    train(
        config=config, predictor_num=predictor_num,
        save_path='./eval/split1_added_full_my_bayesian/',
        load_path='./weights/kitti_split1_added_full_my_bayesian/',
        data_path='./data/kitti_split1_added_full_bayesian',
        batch_size=32, image_size=(1216, 448),
        is_bayesian=False,
    )



