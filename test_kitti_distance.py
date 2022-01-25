import numpy as np
import time, sys, os, scipy.linalg

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

def eval_interval(save_path = None):
    localXYZ_gt_list = np.load(os.path.join(save_path, 'gt.npy'))
    localXYZ_estimated_list = np.load(os.path.join(save_path, 'estimated.npy'))
    sc_gt_list = np.load(os.path.join(save_path, 'sc_gt.npy'))
    sc_pr_list = np.load(os.path.join(save_path, 'sc_pr.npy'))
    txt = ''

    indices_list = []
    indices_list.append((0 < localXYZ_gt_list[..., -1]) & (localXYZ_gt_list[..., -1] <= 10))
    indices_list.append((10 < localXYZ_gt_list[..., -1]) & (localXYZ_gt_list[..., -1] <= 20))
    indices_list.append((20 < localXYZ_gt_list[..., -1]) & (localXYZ_gt_list[..., -1] <= 30))
    indices_list.append((30 < localXYZ_gt_list[..., -1]) & (localXYZ_gt_list[..., -1] <= 40))
    indices_list.append((40 < localXYZ_gt_list[..., -1]) & (localXYZ_gt_list[..., -1] <= 50))
    indices_list.append((50 < localXYZ_gt_list[..., -1]))

    for indices in indices_list:
        localXYZ_gt_list_curr = localXYZ_gt_list[indices]
        localXYZ_estimated_list_curr = localXYZ_estimated_list[indices]
        sc_gt_list_curr = sc_gt_list[indices]
        sc_pr_list_curr = sc_pr_list[indices]
        print(len(localXYZ_gt_list_curr))
        acc, ABS_rel, SQR_rel, RMSE_log, RMSE_lin, error_dist = xyzEval(gt=localXYZ_gt_list_curr, pred=localXYZ_estimated_list_curr)
        txt += '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(acc[0], acc[1], acc[2], ABS_rel, SQR_rel, RMSE_log, RMSE_lin[2])
        acc_4pi, acc_6pi, med_err = SCEval(sc_gt_list=sc_gt_list_curr, sc_pred_list=sc_pr_list_curr)
        txt += '{:.4f} {:.4f} {:.4f}\n'.format(acc_4pi, acc_6pi, med_err)
    with open(os.path.join(save_path, 'eval_interval.txt'), 'w') as fp:
        fp.write(txt)

if __name__ == '__main__':
    # eval_interval(save_path='./eval/split1_org/')
    # eval_interval(save_path='./eval/split1_added_full/')
    # eval_interval(save_path='./eval/split1_added_full_my/')
    # eval_interval(save_path='./eval/split1_added_full_bayesian/')
    eval_interval(save_path='./eval/split1_added_full_my_bayesian/')
    # eval_interval(save_path='./eval/split2_org/')
    # eval_interval(save_path='./eval/split2_added_full/')
    # eval_interval(save_path='./eval/split2_added_full_my/')
    # eval_interval(save_path='./eval/split2_added_full_bayesian/')
    # eval_interval(save_path='./eval/split2_added_full_my_bayesian/')



