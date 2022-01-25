import numpy as np

import src.net_core.darknet as darknet
import src.net_core.autoencoder3D as ae3D
# import src.net_core.priornet as priornet
from src.module.function import *
import src.visualizer.visualizer_ as visualizer
import cv2

config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num':5,
        'bbox2D_dim':4, 'bbox3D_dim':3, 'orientation_dim':3,
        'inst_dim':10, 'z_inst_dim':16,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : 5*(1+4+3+(2*3+3)+2*16),
        'filter_num_list':[1024,1024,1024,1024],
        'filter_size_list':[3,3,3,1],
        'activation':'elu',
    },
    'decoder':{
        'name':'docoder',
        'input_dim' : 16,
        'output_shape':[64,64,64,1],
        'filter_num_list':[512,256,128,64,1],
        'filter_size_list':[4,4,4,4,4],
        'strides_list':[1,2,2,2,2],
        'activation':'elu',
        'final_activation':'sigmoid'
    },
    # 'prior' : {
    #     'name' : 'priornet',
    #     'input_dim' : 10,  # class num (one-hot vector)
    #     'unit_num_list' : [64, 32, 16],
    #     'core_activation' : 'elu',
    #     'const_log_var' : 0.0,
    # }
}

class nolboXYZEval(object):
    def __init__(self,
                 nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 is_bayesian=False
                 ):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        # self._dec_str = nolbo_structure['decoder']
        # self._prior_str = nolbo_structure['prior']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone
        self._is_bayesian = is_bayesian

        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        #             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        #     except RuntimeError as e:
        #         print(e)

        self._buildModel()

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling=None, activation=self._enc_head_str['activation'])
        # ==============set decoder3D
        # self._decoder = ae3D.decoder3D(structure=self._dec_str)
        # self._priornet = priornet.priornet(structure=self._prior_str)
        print('done')

    def _getOffset(self, batchSize, grid_size, pred_num):
        # grid_size_col : gridsize[0]
        # grid_size_row : gridsize[1]
        grid_size_col, grid_size_row = grid_size
        offsetX = np.transpose(np.reshape(
            np.array([np.arange(grid_size_col)] * grid_size_row * pred_num)
            , (pred_num, grid_size_row, grid_size_col)
        ), (1, 2, 0))
        offsetX = np.tile(np.reshape(offsetX, (1, grid_size_row, grid_size_col, pred_num)),
                          [batchSize, 1, 1, 1])
        offsetY = np.transpose(np.reshape(
            np.array([np.arange(grid_size_row)] * grid_size_col * pred_num)
            , (pred_num, grid_size_col, grid_size_row)
        ), (2, 1, 0))
        offsetY = np.tile(np.reshape(offsetY, (1, grid_size_row, grid_size_col, pred_num)),
                          [batchSize, 1, 1, 1])
        return offsetX.astype('float32'), offsetY.astype('float32')

    def getEval(self, inputImages, anchor_bbox3D, anchor_z,
                objnessImages_gt, bbox3DImages_gt, localXYZImages_gt,
                sin_gt, cos_gt, imageSizeImages, P2Images, P2InvImages,
                image_reduced=32, is_exp=False):
        # t = time.time()
        if inputImages.shape[-1] != 3:
            # if gray image
            input_images = np.stack([inputImages, inputImages, inputImages], axis=-1)
        if inputImages.ndim != 4:
            input_images = np.stack([inputImages], axis=0)
        batchSize, inputImgRow, inputImgCol, _ = inputImages.shape
        self._grid_size = [int(inputImgCol / image_reduced), int(inputImgRow / image_reduced)]
        grid_col, grid_row, predictor_num = self._grid_size[0], self._grid_size[1], self._enc_backbone_str[
            'predictor_num']
        self._image_size_org = tf.convert_to_tensor(imageSizeImages)
        self._offset_x, self._offset_y = self._getOffset(
            batchSize=batchSize, grid_size=self._grid_size, pred_num=predictor_num)
        self._P2_gt = tf.convert_to_tensor(P2Images.astype('float32'))
        self._P2_inv_gt = tf.convert_to_tensor(P2InvImages.astype('float32'))
        self._anchor_bbox3D = tf.convert_to_tensor(anchor_bbox3D.astype('float32'))
        self._anchor_z = tf.convert_to_tensor(anchor_z.astype('float32'))
        self._exp = is_exp

        self._input_images = inputImages / 255.
        self._enc_output = self._encoder_backbone(self._input_images, training=False)
        self._enc_output = self._encoder_head(self._enc_output, training=False)
        self._encOutPartitioning()
        self._calcXYZ()
        self._calcBbox3Dand2D()
        # print(1, time.time()-t)
        # t = time.time()
        self._objness = np.array(self._objness)
        self._bbox3D8Points = np.array(self._bbox3D8Points)
        self._R = np.array(self._R)
        self._bbox2D_dim = np.array(self._bbox2D_dim)
        self._bbox3D_dim = np.array(self._bbox3D_dim)
        self._localXYZ = np.array(self._localXYZ)
        self._sin = np.array(self._sin)
        self._cos = np.array(self._cos)
        self._ry = np.array(self._ry)
        self._alpha = np.array(self._alpha)

        # self._bbox3D8Points = np.array(self._bbox3D8Points)
        # self._bbox2D_dim = np.array(self._bbox2D_dim)


        def clip(subjectPolygon, clipPolygon):
            # Sutherland-Hodgman_polygon_clipping
            # arrange polygon in anti-clockwise direction
            def inside(p):
                return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

            def computeIntersection():
                dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
                dp = [s[0] - e[0], s[1] - e[1]]
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = s[0] * e[1] - s[1] * e[0]
                n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
                return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

            outputList = subjectPolygon
            cp1 = clipPolygon[-1]

            for clipVertex in clipPolygon:
                cp2 = clipVertex
                inputList = outputList
                outputList = []
                try:
                    s = inputList[-1]

                    for subjectVertex in inputList:
                        e = subjectVertex
                        if inside(e):
                            if not inside(s):
                                outputList.append(computeIntersection())
                            outputList.append(e)
                        elif inside(s):
                            outputList.append(computeIntersection())
                        s = e
                    cp1 = cp2
                except:
                    return None
            return (outputList)

        def polygonArea(polygon):
            # Shoelace formula
            if polygon is not None:
                num_point = len(polygon)
                area = 0.
                for i in range(num_point):
                    if i + 1 < num_point:
                        area += (polygon[i][0] * polygon[i + 1][1]) - (polygon[i + 1][0] * polygon[i][1])
                    else:
                        area += (polygon[i][0] * polygon[0][1]) - (polygon[0][0] * polygon[i][1])
                if area < 0.:
                    area = -1. * area
                area = 1. / 2. * area
                return area
            else:
                return 0.

        def getIOU3D(pose_src, lhw_src, pose_target, lhw_target):
            def getpoly_anticlock(pose, lhw):
                l, h, w = lhw / 2.
                poly = np.array([[l, -l, -l, l, l, -l, -l, l],
                                 [h, h, h, h, -h, -h, -h, -h],
                                 [w, w, -w, -w, w, w, -w, -w],
                                 [1., 1., 1., 1., 1., 1., 1., 1.]])  # (4, 8)
                poly = np.matmul(pose, poly)  # (4,8)
                lw = poly[[0, 2], 0:4]  # (2, 4)
                lw = np.transpose(lw)  # (4, 2)
                h_max, h_min = np.max(poly[1, :]), np.min(poly[1, :])
                return lw, np.array([h_max, h_min])

            poly_src, h_src = getpoly_anticlock(pose_src, lhw_src)
            h_src_max, h_src_min = h_src
            poly_target, h_target = getpoly_anticlock(pose_target, lhw_target)
            h_target_max, h_target_min = h_target
            polygon = clip(poly_src, poly_target)
            area = polygonArea(polygon)
            h = max(min(h_src_max, h_target_max) - max(h_target_min, h_target_min), 0.)
            volume = area * h
            volume_src = lhw_src[0] * lhw_src[2] * lhw_src[1]
            volume_target = lhw_target[0] * lhw_target[2] * lhw_target[1]
            iou = max(volume / (volume_src + volume_target - volume), 0.)
            if 1. < iou or iou < 0.:
                iou = 0.
            return iou

        def getIoUArgMax(xyz_gt, sin_gt, cos_gt, lhw_gt, xyz_pred_list, sin_pred_list, cos_pred_list, lhw_pred_list):
            def getPose(xyz, sin, cos):
                P = np.array([[cos, 0., sin, xyz[0]],
                              [0., 1., 0., xyz[1]],
                              [-sin, 0., cos, xyz[2]],
                              [0., 0., 0., 1.]])
                return P

            pose_gt = getPose(xyz=xyz_gt, sin=sin_gt, cos=cos_gt)

            IoU_list = []
            X_list = []
            sin_list = []
            cos_list = []
            for xyz_pred, sin_pred, cos_pred, lhw_pred in zip(xyz_pred_list, sin_pred_list, cos_pred_list, lhw_pred_list):
                pose_pred = getPose(xyz=xyz_pred, sin=sin_pred, cos=cos_pred)
                IoU = getIOU3D(pose_src=pose_gt, lhw_src=lhw_gt, pose_target=pose_pred, lhw_target=lhw_pred)
                IoU_list.append(IoU)
                X_list.append(xyz_pred)
                sin_list.append(sin_pred)
                cos_list.append(cos_pred)
            selected_index = int(np.argmax(IoU_list))
            if IoU_list[selected_index] == 0:
                selected_index = np.argmin(np.sqrt(
                        np.abs(np.square(xyz_gt[-1]) - np.square(xyz_pred_list[..., -1]))
                    ))
            return selected_index, X_list[selected_index], sin_list[selected_index], cos_list[selected_index]

        localXYZ_gt = []
        X_list = []
        sc_gt_list = []
        sc_pr_list = []
        index_list = []
        for i in range(len(inputImages)):
            for gr in range(grid_row):
                for gc in range(grid_col):
                    for obj in range(predictor_num):
                        if objnessImages_gt[i, gr, gc, obj, 0] == 1:
                            index, X, spr, cpr = getIoUArgMax(
                                xyz_gt=localXYZImages_gt[i, gr, gc, obj, :],
                                sin_gt=sin_gt[i, gr, gc, obj, 0],
                                cos_gt=cos_gt[i, gr, gc, obj, 0],
                                lhw_gt=bbox3DImages_gt[i, gr, gc, obj, :],
                                xyz_pred_list=self._localXYZ[i, gr, gc, :, :],
                                sin_pred_list=self._sin[i,gr,gc,:,0],
                                cos_pred_list=self._cos[i,gr,gc,:,0],
                                lhw_pred_list=self._bbox3D_dim[i, gr, gc, :, :],
                            )
                            localXYZ_gt.append(localXYZImages_gt[i, gr, gc, obj, :])
                            X_list.append(X)
                            sgt = sin_gt[i, gr, gc, obj, 0]
                            cgt = cos_gt[i, gr, gc, obj, 0]
                            sc_gt_list.append([sgt, cgt])
                            sc_pr_list.append([spr, cpr])
                            index_list.append(index)
                            # self._localXYZ[i, gr, gc, index, :] = 9999., 9999., 9999.

        localXYZ_gt = np.array(localXYZ_gt)
        X_list = np.array(X_list)
        sc_gt_list = np.array(sc_gt_list)
        sc_pr_list = np.array(sc_pr_list)
        return localXYZ_gt, X_list, sc_gt_list, sc_pr_list

    def _encOutPartitioning(self):
        pr_num = self._enc_backbone_str['predictor_num']
        self._objness, self._bbox2D_xy, self._bbox3D_dim, self._localZ = [], [], [], []
        self._sin, self._cos = [], []
        self._localZ_logvar, self._rad_logvar = [], []
        part_start = 0
        part_end = part_start
        for predIndex in range(pr_num):
            # objectness
            part_end += 1
            self._objness.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox2DXY_dim']
            self._bbox2D_xy.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox3D_dim']
            self._bbox3D_dim.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['localXYZ_dim']
            self._localZ.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            if self._is_bayesian:
                part_end += self._enc_backbone_str['localXYZ_dim']
                self._localZ_logvar.append(self._enc_output[..., part_start:part_end])
                part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._sin.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._cos.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            if self._is_bayesian:
                part_end += self._enc_backbone_str['orientation_dim']
                self._rad_logvar.append(self._enc_output[..., part_start:part_end])
                part_start = part_end
            # print(part_end)
        self._objness = tf.sigmoid(tf.transpose(tf.stack(self._objness), [1, 2, 3, 0, 4]))
        self._bbox2D_xy = tf.sigmoid(tf.transpose(tf.stack(self._bbox2D_xy), [1, 2, 3, 0, 4]))
        self._bbox3D_dim = tf.transpose(tf.stack(self._bbox3D_dim), [1, 2, 3, 0, 4])
        self._bbox3D_dim = tf.exp(self._bbox3D_dim) * self._anchor_bbox3D
        self._localZ = tf.transpose(tf.stack(self._localZ), [1, 2, 3, 0, 4])
        if self._exp:
            self._localZ = tf.exp(self._localZ) * tf.expand_dims(self._anchor_z, axis=-1)
        else:
            self._localZ = self._localZ + tf.expand_dims(self._anchor_z, axis=-1)
        self._sin = tf.tanh(tf.transpose(tf.stack(self._sin), [1, 2, 3, 0, 4]))
        self._cos = tf.tanh(tf.transpose(tf.stack(self._cos), [1, 2, 3, 0, 4]))

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))
    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))
    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)

    def _matmul3x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2]
        # return tf.stack([c0, c1, c2], axis=-1)
        c = tf.reduce_sum(a*tf.expand_dims(b, -2), axis=-1)
        return c

    def _matmul4x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2] + a[..., 0, 3] * b[..., 3]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2] + a[..., 1, 3] * b[..., 3]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2] + a[..., 2, 3] * b[..., 3]
        # c3 = a[..., 3, 0] * b[..., 0] + a[..., 3, 1] * b[..., 1] + a[..., 3, 2] * b[..., 2] + a[..., 3, 3] * b[..., 3]
        # return tf.stack([c0, c1, c2, c3], axis=-1)
        c = tf.reduce_sum(a*tf.expand_dims(b, -2), axis=-1)
        return c

    def _get3DBboxAnd2DPorj(self, projmat, R, t, lhw):
        # projmat : (batch, gridrow, girdcol, pred, 4x4)
        # R : (batch, gridrow, gridcol, pred, 3x3)
        # t : (batch, gridrow, gridcol, pred, 3)
        # lhw : (batch, gridrow, gridcol, pred, 3)
        dx, dy, dz = -lhw[..., 0] / 2., -lhw[..., 1] / 2., -lhw[..., 2] / 2.
        dxdydz = []
        for i in range(2):
            dy = -1. * dy
            for j in range(2):
                dx = -1. * dx
                for k in range(2):  # [x,y,z], [
                    dz = -1. * dz
                    dxdydz.append(tf.stack([dx, dy, dz], axis=-1))  # (8, b,gr,gc,pr,3)
        dxdydz = tf.transpose(tf.stack(dxdydz), [1, 2, 3, 4, 0, 5])  # (b,gr,gc,pr,8,3)
        R_tile = tf.transpose(tf.stack([R] * 8), [1, 2, 3, 4, 0, 5, 6])
        t_tile = tf.transpose(tf.stack([t] * 8), [1, 2, 3, 4, 0, 5])
        bbox3D8Points = self._matmul3x1(R_tile, dxdydz) + t_tile  # (b,gr,gc,pr,8,3)
        x_4d = tf.concat([bbox3D8Points, tf.expand_dims(tf.ones_like(dxdydz[..., 0]), axis=-1)], axis=-1)  # (b,gr,gc,pr,8,4)
        projmat_tile = tf.transpose(tf.stack([projmat] * 8), [1, 2, 3, 4, 0, 5, 6])
        bbox3D8PointsProj = self._matmul4x1(projmat_tile, x_4d)
        bbox3D8PointsProj = bbox3D8PointsProj[..., :2] / (tf.expand_dims(bbox3D8PointsProj[..., 2], axis=-1) + 1e-9)
        # print(bbox3D8PointsProj.shape)
        # select proj point
        x1 = tf.reduce_min(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size_org[..., 0]  # (b,gr,gc,pr)
        x2 = tf.reduce_max(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size_org[..., 0]
        y1 = tf.reduce_min(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size_org[..., 1]
        y2 = tf.reduce_max(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size_org[..., 1]
        # print(x1.shape)
        return bbox3D8Points, tf.stack([x1, y1, x2, y2], axis=-1)  # (b,gr,gc,pr,4)

    def _calcXYZ(self):
        len_grid_x, len_grid_y = tf.cast(tf.shape(self._offset_x)[2], tf.float32), tf.cast(tf.shape(self._offset_x)[1], tf.float32)
        # image_size : (row, col)
        objCenter2D_xz = (self._bbox2D_xy[..., 0] + self._offset_x) / len_grid_x * self._image_size_org[..., 0] * self._localZ[..., 0]
        objCenter2D_yz = (self._bbox2D_xy[..., 1] + self._offset_y) / len_grid_y * self._image_size_org[..., 1] * self._localZ[..., 0]
        objCenter2D_xyz = tf.stack([objCenter2D_xz, objCenter2D_yz, self._localZ[..., 0], tf.ones_like(self._localZ[..., 0])], axis=-1)
        self._localXYZ = self._matmul4x1(self._P2_inv_gt, objCenter2D_xyz)[..., 0:3]

    def _calcBbox3Dand2D(self):
        b, gr, gc, pr, _ = tf.shape(self._cos)
        zx_norm = tf.sqrt(tf.square(self._localXYZ[..., -1]) + tf.square(self._localXYZ[..., 0]))
        s_ray, c_ray = tf.expand_dims(self._localXYZ[..., 0] / zx_norm, axis=-1), tf.expand_dims(self._localXYZ[..., -1] / zx_norm, axis=-1)
        s_ray, c_ray = tf.constant(s_ray.numpy()), tf.constant(c_ray.numpy())
        self._sin_ry = s_ray * self._cos + c_ray * self._sin
        self._cos_ry = c_ray * self._cos - s_ray * self._sin
        # self._sin_ry, self._cos_ry = self._sin, self._cos
        zero = tf.zeros_like(self._cos_ry)
        one = tf.ones_like(self._cos_ry)
        self._R = tf.reshape(tf.concat([self._cos_ry, zero, self._sin_ry, zero, one, zero, -self._sin_ry, zero, self._cos_ry], axis=-1), [b, gr, gc, pr, 3, 3])
        self._bbox3D8Points, self._bbox2D_dim = self._get3DBboxAnd2DPorj(self._P2_gt, self._R, self._localXYZ, tf.constant(self._bbox3D_dim.numpy()))
        # self._bbox3D8Points
        # self._bbox2D_dim
        #
        self._ry = tf.atan2(self._sin_ry, self._cos_ry)
        self._alpha = tf.atan2(self._sin, self._cos)
        # self._bbox3D8Points
        # self._bbox2D_dim
        #

        #final output
        # self._localXYZ
        # self._R
        # self._bbox3D8Points
        # self._bbox2D_dim_normalized


class nolbo_test(object):
    def __init__(self, nolbo_structure):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._rad_var = (15.0 / 180.0 * 3.141593) ** 2
        self._category_num = nolbo_structure['category_num']
        self._buildModel()

    def _buildModel(self):
        print('build Models...')
        self._encoder_core = darknet.Darknet19(name=self._enc_backbone_str['name'], activation='lrelu')
        self._encoder_core_head = darknet.Darknet19_head2D(name=self._enc_backbone_str['name'] + '_head',
                                                           activation='lrelu')
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=[None, None, 1024],
                                            output_dim=self._enc_head_str['output_dim'],
                                            last_pooling=None, activation=self._enc_head_str['activation'])
        # #==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        self._classifier = darknet.classifier(name=None, input_shape=[64, ], output_dim=self._category_num,
                                              activation='relu')
        print('done')

    def _getOffset(self, batchSize, grid_size, pred_num):
        # grid_size_col : gridsize[0]
        # grid_size_row : gridsize[1]
        grid_size_col, grid_size_row = grid_size
        offsetX = np.transpose(np.reshape(
            np.array([np.arange(grid_size_col)] * grid_size_row * pred_num)
            , (pred_num, grid_size_row, grid_size_col)
        ), (1, 2, 0))
        offsetX = np.tile(np.reshape(offsetX, (1, grid_size_row, grid_size_col, pred_num)),
                          [batchSize, 1, 1, 1])
        offsetY = np.transpose(np.reshape(
            np.array([np.arange(grid_size_row)] * grid_size_col * pred_num)
            , (pred_num, grid_size_col, grid_size_row)
        ), (2, 1, 0))
        offsetY = np.tile(np.reshape(offsetY, (1, grid_size_row, grid_size_col, pred_num)),
                          [batchSize, 1, 1, 1])
        return offsetX.astype('float32'), offsetY.astype('float32')

    def getPred(self,
                input_images, image_mean, image_std, P2_gt, P2_inv_gt, image_size_org,  # (row, col)
                anchor_bbox3D, anchor_z,
                obj_thresh=0.5, IOU_thresh=0.3, get_shape=False,
                image_reduced=32):
        # t = time.time()
        if input_images.shape[-1] != 3:
            # if gray image
            input_images = np.stack([input_images, input_images, input_images], axis=-1)
        if input_images.ndim != 4:
            input_images = np.stack([input_images], axis=0)
        batchSize, inputImgRow, inputImgCol, _ = input_images.shape
        self._grid_size = [int(inputImgCol / image_reduced), int(inputImgRow / image_reduced)]
        grid_col, grid_row, predictor_num = self._grid_size[0], self._grid_size[1], self._enc_backbone_str['predictor_num']
        self._image_size_org = np.zeros((batchSize, grid_row, grid_col, predictor_num, 2))
        self._image_size_org[0, :, :, :, :] = image_size_org
        self._offset_x, self._offset_y = self._getOffset(
            batchSize=batchSize, grid_size=self._grid_size, pred_num=predictor_num)
        self._P2_gt = np.zeros((batchSize, grid_row, grid_col, predictor_num, 4, 4))
        self._P2_inv_gt = np.zeros_like(self._P2_gt)
        self._P2_gt[0, :, :, :, :, :] = P2_gt
        self._P2_inv_gt[0, :, :, :, :, :] = P2_inv_gt
        self._P2_gt = tf.convert_to_tensor(self._P2_gt.astype('float32'))
        self._P2_inv_gt = tf.convert_to_tensor(self._P2_inv_gt.astype('float32'))
        self._anchor_bbox3D = tf.convert_to_tensor(anchor_bbox3D.astype('float32'))
        self._anchor_z = tf.convert_to_tensor(anchor_z.astype('float32'))

        self._input_images = (input_images - image_mean)/image_std
        self._enc_output = self._encoder_head(
            self._encoder_core_head(
                self._encoder_core(self._input_images, training=False)
                , training=False)
            , training=False)
        self._encOutPartitioning()
        self._calcXYZ()
        self._calcBbox3Dand2D()
        # print(1, time.time()-t)
        # t = time.time()
        self._objness = np.array(self._objness)
        self._bbox3D8Points = np.array(self._bbox3D8Points)
        self._R = np.array(self._R)
        self._bbox2D_dim = np.array(self._bbox2D_dim)
        self._bbox3D_dim = np.array(self._bbox3D_dim)
        self._localXYZ = np.array(self._localXYZ)
        self._sin, self._cos = np.array(self._sin), np.array(self._cos)
        self._ry = np.array(self._ry)
        self._alpha = np.array(self._alpha)
        self._latent_mean = np.array(self._latent_mean)
        self._latent_log_var = np.array(self._latent_log_var)
        # self._bbox3D8Points = np.array(self._bbox3D8Points)
        # self._bbox2D_dim = np.array(self._bbox2D_dim)

        Rdet = self._sin * self._sin + self._cos * self._cos
        selected_indices = (self._objness[..., 0] > obj_thresh) & ((Rdet[..., 0] > 0.90) & (Rdet[..., 0] < 1.1))
        objness_selected = self._objness[selected_indices]
        localXYZ_selected = self._localXYZ[selected_indices]
        R_selected = self._R[selected_indices]
        bbox3D8Points_selected = self._bbox3D8Points[selected_indices]
        bbox2D4Points_selected = self._bbox2D_dim[selected_indices]
        bbox3DDim_selected = self._bbox3D_dim[selected_indices]
        ry_selected = self._ry[selected_indices]
        alpha_selected = self._alpha[selected_indices]
        latent_mean_selected = self._latent_mean[selected_indices]
        latent_log_var_selected = self._latent_log_var[selected_indices]

        # z_inversed = np.expand_dims(1./localXYZ_selected[..., -1], axis=-1)
        # bbox_and_z_inversed = np.concatenate([bbox2D4Points_selected, z_inversed], axis=-1)
        bbox_and_objness = np.concatenate([bbox2D4Points_selected, objness_selected], axis=-1)
        selected_nms = nonMaximumSuppresion(boxes=bbox_and_objness, IOUThreshold=IOU_thresh)
        # selected_nms = self._nonMaximumSuppression(self._objness[selected_indices, 0], bbox3D8Points_selected,
        #                                            localXYZ_selected,
        #                                            bbox3DDim_selected,
        #                                            IOUThreshold=IOU_thresh)
        objness_selected = objness_selected[selected_nms]
        localXYZ_selected = localXYZ_selected[selected_nms]
        R_selected = R_selected[selected_nms]
        bbox3D8Points_selected = bbox3D8Points_selected[selected_nms]
        bbox2D4Points_selected = bbox2D4Points_selected[selected_nms]
        bbox3DDim_selected = bbox3DDim_selected[selected_nms]
        ry_selected = ry_selected[selected_nms]
        alpha_selected = alpha_selected[selected_nms]
        latent_mean_selected = latent_mean_selected[selected_nms]
        latent_log_var_selected = latent_log_var_selected[selected_nms]
        z_selected = []
        category_label = []
        if len(latent_mean_selected)>0:
            latent_mean_selected, latent_log_var_selected = tf.convert_to_tensor(latent_mean_selected), tf.convert_to_tensor(latent_log_var_selected)
            z_selected = sampling(mu=latent_mean_selected, logVar=latent_log_var_selected)
            category_label = self._classifier(z_selected, training=False)
            category_label = np.argmax(np.array(category_label), axis=-1)

        if len(localXYZ_selected) > 0:
            bbox2D4Points_selected[..., 0] = bbox2D4Points_selected[..., 0] * image_size_org[0]
            bbox2D4Points_selected[..., 1] = bbox2D4Points_selected[..., 1] * image_size_org[1]
            bbox2D4Points_selected[..., 2] = bbox2D4Points_selected[..., 2] * image_size_org[0]
            bbox2D4Points_selected[..., 3] = bbox2D4Points_selected[..., 3] * image_size_org[1]
            pose_padding = np.zeros((len(R_selected), 1, 4))
            pose_padding[:, :, :] = np.array([0., 0., 0., 1.])
            pose_selected = np.concatenate(
                [np.concatenate([R_selected, np.expand_dims(localXYZ_selected, axis=-1)], axis=-1), pose_padding],
                axis=-2)
        else:
            pose_selected = np.array([])

        objPoints_selected = []
        if get_shape:
            if len(z_selected) > 0:
                # is_car = category_label == 0
                # z_car = z_selected[is_car]
                # shape = self._decoder(z_car, training=False)
                shape_selected = np.reshape(np.array(self._decoder(z_selected, training=False)), [-1, 64,64,64])
                shape_selected = np.array(shape_selected)
                for shape, bbox3DDim, objPose in zip(shape_selected, bbox3DDim_selected, pose_selected):
                    b3l, b3h, b3w = bbox3DDim
                    objPoints = visualizer.objRescaleTransform(shape, b3h, b3w, b3l, objPose)
                    objPoints_selected.append(objPoints)


        return pose_selected, bbox3DDim_selected, bbox3D8Points_selected, bbox2D4Points_selected, ry_selected, alpha_selected, objness_selected, category_label, objPoints_selected

    def _nonMaximumSuppression(self, objness, bbox3D8Points, localXYZ, bbox3DDim, IOUThreshold):

        def clip(subjectPolygon, clipPolygon):
            # Sutherland-Hodgman_polygon_clipping
            # arrange polygon in anti-clockwise direction
            def inside(p):
                return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

            def computeIntersection():
                dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
                dp = [s[0] - e[0], s[1] - e[1]]
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = s[0] * e[1] - s[1] * e[0]
                n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
                return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

            outputList = subjectPolygon
            cp1 = clipPolygon[-1]

            for clipVertex in clipPolygon:
                cp2 = clipVertex
                inputList = outputList
                outputList = []
                try:
                    s = inputList[-1]

                    for subjectVertex in inputList:
                        e = subjectVertex
                        if inside(e):
                            if not inside(s):
                                outputList.append(computeIntersection())
                            outputList.append(e)
                        elif inside(s):
                            outputList.append(computeIntersection())
                        s = e
                    cp1 = cp2
                except:
                    return None
            return (outputList)

        def polygonArea(polygon):
            # Shoelace formula
            num_point = len(polygon)
            area = 0.
            for i in range(num_point):
                if i + 1 < num_point:
                    area += (polygon[i][0] * polygon[i + 1][1]) - (polygon[i + 1][0] * polygon[i][1])
                else:
                    area += (polygon[i][0] * polygon[0][1]) - (polygon[0][0] * polygon[i][1])
            if area < 0.:
                area = -1. * area
            area = 1. / 2. * area
            return area

        pickedBoxIdx = []
        if len(objness) > 0:
            y_mins = np.min(bbox3D8Points[..., 1], axis=-1)
            y_maxs = np.max(bbox3D8Points[..., 1], axis=-1)
            # (x,z; x,-z; -x,z; -x,-z) -> (x,z; -x,z; -x,-z; x,-z)
            # print(bbox3D8Points.shape)
            xzs = bbox3D8Points[:, 0:4, [0, 2, 1]]
            xzs = xzs[:, [0, 2, 3, 1], 0:2]
            # print(xzs.shape)
            # xzs = xzs[:, [0,2,3,1], [0,2,1]]
            # xzs = xzs[..., 0:2]

            boxIdxs = np.argsort(localXYZ[..., -1])
            # print('nms')
            while len(boxIdxs) > 0:
                # print(boxIdxs)
                sortedBoxLastIdx = len(boxIdxs) - 1
                boxIdxCurr = boxIdxs[sortedBoxLastIdx]
                xz_curr = xzs[boxIdxCurr]
                y_max_curr = y_maxs[boxIdxCurr]
                y_min_curr = y_mins[boxIdxCurr]
                vol_curr = np.prod(bbox3DDim[boxIdxCurr])
                pickedBoxIdx.append(boxIdxCurr)
                IOUs = []
                for y_min, y_max, xz, bboxlhw in zip(y_mins[boxIdxs[:sortedBoxLastIdx]],
                                                     y_maxs[boxIdxs[:sortedBoxLastIdx]],
                                                     xzs[boxIdxs[:sortedBoxLastIdx]],
                                                     bbox3DDim[boxIdxs[:sortedBoxLastIdx]]):
                    cliped = clip(xz, xz_curr)
                    if cliped is None:
                        # print('None')
                        # print(xz)
                        # print(xzs[-1])
                        area_intersection = 0.
                    else:
                        area_intersection = polygonArea(cliped)
                    h_max = np.min([y_max, y_max_curr])
                    h_min = np.max([y_min, y_min_curr])
                    h = np.max([h_max - h_min, 0.])
                    vol_intersection = area_intersection * h
                    vol = np.prod(bboxlhw)
                    IOU3D = vol_intersection / (vol_curr + vol - vol_intersection)
                    IOUs.append(IOU3D)
                    # if IOU3D>IOUThreshold:
                    #     print(xz)
                    #     print(xzs[-1])
                    #     print(IOU3D)
                IOUs = np.array(IOUs)
                boxIdxs = np.delete(boxIdxs, np.concatenate(([sortedBoxLastIdx], np.where(IOUs > IOUThreshold)[0])))
        return pickedBoxIdx

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_core.load_weights(os.path.join(load_path, file_name))
    def loadEncoderHead(self, load_path):
        file_name = self._enc_backbone_str['name'] + '_head'
        self._encoder_core_head.load_weights(os.path.join(load_path, file_name))
        file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))
    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)
    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))
    # def loadPriornet(self, load_path):
    #     file_name = self._prior_str['name']
    #     self._priornet_car.load_weights(os.path.join(load_path, file_name))
    def loadClassifier(self, load_path):
        file_name = 'classifier'
        self._classifier.load_weights(os.path.join(load_path, file_name))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        # self.loadPriornet(load_path=load_path)
        self.loadClassifier(load_path=load_path)

    def _encOutPartitioning(self):
        pr_num = self._enc_backbone_str['predictor_num']
        self._objness, self._bbox2D_xy, self._bbox3D_dim, self._localZ = [], [], [], []
        self._latent_mean, self._latent_log_var = [], []
        self._sin, self._cos = [], []
        part_start = 0
        part_end = part_start
        for predIndex in range(pr_num):
            # objectness
            part_end += 1
            self._objness.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox2DXY_dim']
            self._bbox2D_xy.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox3D_dim']
            self._bbox3D_dim.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['localXYZ_dim']
            self._localZ.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['latent_dim']
            self._latent_mean.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['latent_dim']
            self._latent_log_var.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._sin.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._cos.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            # part_end += self._enc_backbone_str['orientation_dim']
            # self._rad_log_var.append(self._enc_output[..., part_start:part_end])
            # part_start = part_end
            # print(part_end)
        self._objness = tf.sigmoid(tf.transpose(tf.stack(self._objness), [1, 2, 3, 0, 4]))
        self._bbox2D_xy = tf.sigmoid(tf.transpose(tf.stack(self._bbox2D_xy), [1, 2, 3, 0, 4]))
        self._bbox3D_dim = tf.transpose(tf.stack(self._bbox3D_dim), [1, 2, 3, 0, 4])
        self._bbox3D_dim = tf.clip_by_value(self._bbox3D_dim, clip_value_min=-3.0, clip_value_max=3.0)
        self._bbox3D_dim = tf.exp(self._bbox3D_dim) * self._anchor_bbox3D
        # print(self._bbox3D_dim.shape)
        self._localZ = tf.transpose(tf.stack(self._localZ), [1, 2, 3, 0, 4])
        self._localZ = self._localZ + tf.expand_dims(self._anchor_z, axis=-1)
        self._latent_mean = tf.transpose(tf.stack(self._latent_mean), [1, 2, 3, 0, 4])
        self._latent_log_var = tf.transpose(tf.stack(self._latent_log_var), [1, 2, 3, 0, 4])
        self._sin = tf.tanh(tf.transpose(tf.stack(self._sin), [1, 2, 3, 0, 4]))
        self._cos = tf.tanh(tf.transpose(tf.stack(self._cos), [1, 2, 3, 0, 4]))

    def _matmul3x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2]
        # return tf.stack([c0, c1, c2], axis=-1)
        c = tf.reduce_sum(a*tf.expand_dims(b, -2), axis=-1)
        return c

    def _matmul4x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2] + a[..., 0, 3] * b[..., 3]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2] + a[..., 1, 3] * b[..., 3]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2] + a[..., 2, 3] * b[..., 3]
        # c3 = a[..., 3, 0] * b[..., 0] + a[..., 3, 1] * b[..., 1] + a[..., 3, 2] * b[..., 2] + a[..., 3, 3] * b[..., 3]
        # return tf.stack([c0, c1, c2, c3], axis=-1)
        c = tf.reduce_sum(a*tf.expand_dims(b, -2), axis=-1)
        return c

    def _get3DBboxAnd2DPorj(self, projmat, R, t, lhw):
        # projmat : (batch, gridrow, girdcol, pred, 4x4)
        # R : (batch, gridrow, gridcol, pred, 3x3)
        # t : (batch, gridrow, gridcol, pred, 3)
        # lhw : (batch, gridrow, gridcol, pred, 3)
        dx, dy, dz = -lhw[..., 0] / 2., -lhw[..., 1] / 2., -lhw[..., 2] / 2.
        dxdydz = []
        for i in range(2):
            dy = -1. * dy
            for j in range(2):
                dx = -1. * dx
                for k in range(2):  # [x,y,z], [
                    dz = -1. * dz
                    dxdydz.append(tf.stack([dx, dy, dz], axis=-1))  # (8, b,gr,gc,pr,3)
        dxdydz = tf.transpose(tf.stack(dxdydz), [1, 2, 3, 4, 0, 5])  # (b,gr,gc,pr,8,3)
        R_tile = tf.transpose(tf.stack([R] * 8), [1, 2, 3, 4, 0, 5, 6])
        t_tile = tf.transpose(tf.stack([t] * 8), [1, 2, 3, 4, 0, 5])
        bbox3D8Points = self._matmul3x1(R_tile, dxdydz) + t_tile  # (b,gr,gc,pr,8,3)
        x_4d = tf.concat([bbox3D8Points, tf.expand_dims(tf.ones_like(dxdydz[..., 0]), axis=-1)], axis=-1)  # (b,gr,gc,pr,8,4)
        projmat_tile = tf.transpose(tf.stack([projmat] * 8), [1, 2, 3, 4, 0, 5, 6])
        bbox3D8PointsProj = self._matmul4x1(projmat_tile, x_4d)
        bbox3D8PointsProj = bbox3D8PointsProj[..., :2] / (tf.expand_dims(bbox3D8PointsProj[..., 2], axis=-1) + 1e-9)
        # print(bbox3D8PointsProj.shape)
        # select proj point
        x1 = tf.reduce_min(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size_org[..., 0]  # (b,gr,gc,pr)
        x2 = tf.reduce_max(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size_org[..., 0]
        y1 = tf.reduce_min(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size_org[..., 1]
        y2 = tf.reduce_max(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size_org[..., 1]
        # print(x1.shape)
        return bbox3D8Points, tf.stack([x1, y1, x2, y2], axis=-1)  # (b,gr,gc,pr,4)

    def _calcXYZ(self):
        len_grid_x, len_grid_y = tf.cast(tf.shape(self._offset_x)[2], tf.float32), tf.cast(tf.shape(self._offset_x)[1], tf.float32)
        # image_size : (row, col)
        objCenter2D_xz = (self._bbox2D_xy[..., 0] + self._offset_x) / len_grid_x * self._image_size_org[..., 0] * self._localZ[..., 0]
        objCenter2D_yz = (self._bbox2D_xy[..., 1] + self._offset_y) / len_grid_y * self._image_size_org[..., 1] * self._localZ[..., 0]
        objCenter2D_xyz = tf.stack([objCenter2D_xz, objCenter2D_yz, self._localZ[..., 0], tf.ones_like(self._localZ[..., 0])], axis=-1)
        self._localXYZ = self._matmul4x1(self._P2_inv_gt, objCenter2D_xyz)[..., 0:3]

    def _calcBbox3Dand2D(self):
        b, gr, gc, pr, _ = tf.shape(self._cos)
        zx_norm = tf.sqrt(tf.square(self._localXYZ[..., -1]) + tf.square(self._localXYZ[..., 0]))
        s_ray, c_ray = tf.expand_dims(self._localXYZ[..., 0] / zx_norm, axis=-1), tf.expand_dims(self._localXYZ[..., -1] / zx_norm, axis=-1)
        s_ray, c_ray = tf.constant(s_ray.numpy()), tf.constant(c_ray.numpy())
        self._sin_ry = s_ray * self._cos + c_ray * self._sin
        self._cos_ry = c_ray * self._cos - s_ray * self._sin
        # self._sin_ry, self._cos_ry = self._sin, self._cos
        zero = tf.zeros_like(self._cos_ry)
        one = tf.ones_like(self._cos_ry)
        self._R = tf.reshape(tf.concat([self._cos_ry, zero, self._sin_ry, zero, one, zero, -self._sin_ry, zero, self._cos_ry], axis=-1), [b, gr, gc, pr, 3, 3])
        self._bbox3D8Points, self._bbox2D_dim = self._get3DBboxAnd2DPorj(self._P2_gt, self._R, self._localXYZ, tf.constant(self._bbox3D_dim.numpy()))
        # self._bbox3D8Points
        # self._bbox2D_dim
        #
        self._ry = tf.atan2(self._sin_ry, self._cos_ry)
        self._alpha = tf.atan2(self._sin, self._cos)
        # self._bbox3D8Points
        # self._bbox2D_dim
        #

        #final output
        # self._localXYZ
        # self._R
        # self._bbox3D8Points
        # self._bbox2D_dim_normalized


class nolbo_test_bayesian(object):
    def __init__(self,
                 nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 ):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        # self._dec_str = nolbo_structure['decoder']
        # self._prior_str = nolbo_structure['prior']

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone
        self._rad_var = (15.0 / 180.0 * 3.141593) ** 2
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     try:
        #         tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        #             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        #     except RuntimeError as e:
        #         print(e)
        self._buildModel()

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling=None, activation=self._enc_head_str['activation'])
        print('done')

    def _getOffset(self, batchSize, grid_size, pred_num):
        # grid_size_col : gridsize[0]
        # grid_size_row : gridsize[1]
        grid_size_col, grid_size_row = grid_size
        offsetX = np.transpose(np.reshape(
            np.array([np.arange(grid_size_col)] * grid_size_row * pred_num)
            , (pred_num, grid_size_row, grid_size_col)
        ), (1, 2, 0))
        offsetX = np.tile(np.reshape(offsetX, (1, grid_size_row, grid_size_col, pred_num)),
                          [batchSize, 1, 1, 1])
        offsetY = np.transpose(np.reshape(
            np.array([np.arange(grid_size_row)] * grid_size_col * pred_num)
            , (pred_num, grid_size_col, grid_size_row)
        ), (2, 1, 0))
        offsetY = np.tile(np.reshape(offsetY, (1, grid_size_row, grid_size_col, pred_num)),
                          [batchSize, 1, 1, 1])
        return offsetX.astype('float32'), offsetY.astype('float32')

    def getPred(self,
                input_images, P2_gt, P2_inv_gt, image_size_org,  # (row, col)
                anchor_bbox3D, anchor_z,
                obj_thresh=0.5, IOU_thresh=0.3, is_exp=False,
                image_reduced=32):
        # t = time.time()
        if input_images.shape[-1] != 3:
            # if gray image
            input_images = np.stack([input_images, input_images, input_images], axis=-1)
        if input_images.ndim != 4:
            input_images = np.stack([input_images], axis=0)
        batchSize, inputImgRow, inputImgCol, _ = input_images.shape
        self._grid_size = [int(inputImgCol / image_reduced), int(inputImgRow / image_reduced)]
        grid_col, grid_row, predictor_num = self._grid_size[0], self._grid_size[1], self._enc_backbone_str['predictor_num']
        self._image_size_org = np.zeros((batchSize, grid_row, grid_col, predictor_num, 2))
        self._image_size_org[0, :, :, :, :] = image_size_org
        self._offset_x, self._offset_y = self._getOffset(
            batchSize=batchSize, grid_size=self._grid_size, pred_num=predictor_num)
        self._P2_gt = np.zeros((batchSize, grid_row, grid_col, predictor_num, 4, 4))
        self._P2_inv_gt = np.zeros_like(self._P2_gt)
        self._P2_gt[0, :, :, :, :, :] = P2_gt
        self._P2_inv_gt[0, :, :, :, :, :] = P2_inv_gt
        self._P2_gt = tf.convert_to_tensor(self._P2_gt.astype('float32'))
        self._P2_inv_gt = tf.convert_to_tensor(self._P2_inv_gt.astype('float32'))
        self._anchor_bbox3D = tf.convert_to_tensor(anchor_bbox3D.astype('float32'))
        self._anchor_z = tf.convert_to_tensor(anchor_z.astype('float32'))
        self._exp = is_exp

        self._input_images = input_images / 255.
        self._enc_output = self._encoder_backbone(self._input_images, training=False)
        self._enc_output = self._encoder_head(self._enc_output, training=False)
        self._encOutPartitioning()
        self._calcXYZ()
        self._calcBbox3Dand2D()
        # print(1, time.time()-t)
        # t = time.time()
        self._objness = np.array(self._objness)
        self._bbox3D8Points = np.array(self._bbox3D8Points)
        self._R = np.array(self._R)
        self._bbox2D_dim = np.array(self._bbox2D_dim)
        self._bbox3D_dim = np.array(self._bbox3D_dim)
        self._localXYZ = np.array(self._localXYZ)
        self._sin, self._cos = np.array(self._sin), np.array(self._cos)
        self._ry = np.array(self._ry)
        self._alpha = np.array(self._alpha)
        self._localZ_logvar = np.array(self._localZ_logvar)
        # self._bbox3D8Points = np.array(self._bbox3D8Points)
        # self._bbox2D_dim = np.array(self._bbox2D_dim)

        Rdet = self._sin * self._sin + self._cos * self._cos
        selected_indices = (self._objness[..., 0] > obj_thresh) & ((Rdet[..., 0] > 0.90) & (Rdet[..., 0] < 1.1))
        objness_selected = self._objness[selected_indices]
        localXYZ_selected = self._localXYZ[selected_indices]
        R_selected = self._R[selected_indices]
        bbox3D8Points_selected = self._bbox3D8Points[selected_indices]
        bbox2D4Points_selected = self._bbox2D_dim[selected_indices]
        bbox3DDim_selected = self._bbox3D_dim[selected_indices]
        ry_selected = self._ry[selected_indices]
        alpha_selected = self._alpha[selected_indices]
        localZ_sqrtvar_inverse_selected = 1./np.sqrt(np.exp(self._localZ_logvar[selected_indices]))

        # z_inversed = np.expand_dims(1./localXYZ_selected[..., -1], axis=-1)
        # bbox_and_z_inversed = np.concatenate([bbox2D4Points_selected, z_inversed], axis=-1)
        bbox_and_objness = np.concatenate([bbox2D4Points_selected, objness_selected], axis=-1)
        selected_nms = nonMaximumSuppresion(boxes=bbox_and_objness, IOUThreshold=IOU_thresh)
        # selected_nms = self._nonMaximumSuppression(self._objness[selected_indices, 0], bbox3D8Points_selected,
        #                                            localXYZ_selected,
        #                                            bbox3DDim_selected,
        #                                            IOUThreshold=IOU_thresh)
        objness_selected = objness_selected[selected_nms]
        localXYZ_selected = localXYZ_selected[selected_nms]
        R_selected = R_selected[selected_nms]
        bbox3D8Points_selected = bbox3D8Points_selected[selected_nms]
        bbox2D4Points_selected = bbox2D4Points_selected[selected_nms]
        bbox3DDim_selected = bbox3DDim_selected[selected_nms]
        ry_selected = ry_selected[selected_nms]
        alpha_selected = alpha_selected[selected_nms]
        localZ_sqrtvar_inverse_selected = localZ_sqrtvar_inverse_selected[selected_nms]

        if len(localXYZ_selected) > 0:
            bbox2D4Points_selected[..., 0] = bbox2D4Points_selected[..., 0] * image_size_org[0]
            bbox2D4Points_selected[..., 1] = bbox2D4Points_selected[..., 1] * image_size_org[1]
            bbox2D4Points_selected[..., 2] = bbox2D4Points_selected[..., 2] * image_size_org[0]
            bbox2D4Points_selected[..., 3] = bbox2D4Points_selected[..., 3] * image_size_org[1]
            pose_padding = np.zeros((len(R_selected), 1, 4))
            pose_padding[:, :, :] = np.array([0., 0., 0., 1.])
            pose_selected = np.concatenate(
                [np.concatenate([R_selected, np.expand_dims(localXYZ_selected, axis=-1)], axis=-1), pose_padding],
                axis=-2)
        else:
            pose_selected = np.array([])

        return pose_selected, bbox3DDim_selected, bbox3D8Points_selected, bbox2D4Points_selected, ry_selected, alpha_selected, localZ_sqrtvar_inverse_selected

    def _nonMaximumSuppression(self, objness, bbox3D8Points, localXYZ, bbox3DDim, IOUThreshold):

        def clip(subjectPolygon, clipPolygon):
            # Sutherland-Hodgman_polygon_clipping
            # arrange polygon in anti-clockwise direction
            def inside(p):
                return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

            def computeIntersection():
                dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
                dp = [s[0] - e[0], s[1] - e[1]]
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = s[0] * e[1] - s[1] * e[0]
                n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
                return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

            outputList = subjectPolygon
            cp1 = clipPolygon[-1]

            for clipVertex in clipPolygon:
                cp2 = clipVertex
                inputList = outputList
                outputList = []
                try:
                    s = inputList[-1]

                    for subjectVertex in inputList:
                        e = subjectVertex
                        if inside(e):
                            if not inside(s):
                                outputList.append(computeIntersection())
                            outputList.append(e)
                        elif inside(s):
                            outputList.append(computeIntersection())
                        s = e
                    cp1 = cp2
                except:
                    return None
            return (outputList)

        def polygonArea(polygon):
            # Shoelace formula
            num_point = len(polygon)
            area = 0.
            for i in range(num_point):
                if i + 1 < num_point:
                    area += (polygon[i][0] * polygon[i + 1][1]) - (polygon[i + 1][0] * polygon[i][1])
                else:
                    area += (polygon[i][0] * polygon[0][1]) - (polygon[0][0] * polygon[i][1])
            if area < 0.:
                area = -1. * area
            area = 1. / 2. * area
            return area

        pickedBoxIdx = []
        if len(objness) > 0:
            y_mins = np.min(bbox3D8Points[..., 1], axis=-1)
            y_maxs = np.max(bbox3D8Points[..., 1], axis=-1)
            # (x,z; x,-z; -x,z; -x,-z) -> (x,z; -x,z; -x,-z; x,-z)
            # print(bbox3D8Points.shape)
            xzs = bbox3D8Points[:, 0:4, [0, 2, 1]]
            xzs = xzs[:, [0, 2, 3, 1], 0:2]
            # print(xzs.shape)
            # xzs = xzs[:, [0,2,3,1], [0,2,1]]
            # xzs = xzs[..., 0:2]

            boxIdxs = np.argsort(localXYZ[..., -1])
            # print('nms')
            while len(boxIdxs) > 0:
                # print(boxIdxs)
                sortedBoxLastIdx = len(boxIdxs) - 1
                boxIdxCurr = boxIdxs[sortedBoxLastIdx]
                xz_curr = xzs[boxIdxCurr]
                y_max_curr = y_maxs[boxIdxCurr]
                y_min_curr = y_mins[boxIdxCurr]
                vol_curr = np.prod(bbox3DDim[boxIdxCurr])
                pickedBoxIdx.append(boxIdxCurr)
                IOUs = []
                for y_min, y_max, xz, bboxlhw in zip(y_mins[boxIdxs[:sortedBoxLastIdx]],
                                                     y_maxs[boxIdxs[:sortedBoxLastIdx]],
                                                     xzs[boxIdxs[:sortedBoxLastIdx]],
                                                     bbox3DDim[boxIdxs[:sortedBoxLastIdx]]):
                    cliped = clip(xz, xz_curr)
                    if cliped is None:
                        # print('None')
                        # print(xz)
                        # print(xzs[-1])
                        area_intersection = 0.
                    else:
                        area_intersection = polygonArea(cliped)
                    h_max = np.min([y_max, y_max_curr])
                    h_min = np.max([y_min, y_min_curr])
                    h = np.max([h_max - h_min, 0.])
                    vol_intersection = area_intersection * h
                    vol = np.prod(bboxlhw)
                    IOU3D = vol_intersection / (vol_curr + vol - vol_intersection)
                    IOUs.append(IOU3D)
                    # if IOU3D>IOUThreshold:
                    #     print(xz)
                    #     print(xzs[-1])
                    #     print(IOU3D)
                IOUs = np.array(IOUs)
                boxIdxs = np.delete(boxIdxs, np.concatenate(([sortedBoxLastIdx], np.where(IOUs > IOUThreshold)[0])))
        return pickedBoxIdx

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_backbone_str['name']
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))

    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._enc_head_str['name']
        self._encoder_head.load_weights(os.path.join(load_path, file_name))

    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)

    # def loadDecoder(self, load_path, file_name=None):
    #     if file_name == None:
    #         file_name = self._dec_str['name']
    #     self._decoder.load_weights(os.path.join(load_path, file_name))
    # def loadPriornet(self, load_path, file_name=None):
    #     if file_name == None:
    #         file_name = self._prior_str['name']
    #     self._priornet.load_weights(os.path.join(load_path, file_name))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        # self.loadDecoder(load_path=load_path)
        # self.loadPriornet(load_path=load_path)

    def _encOutPartitioning(self):
        pr_num = self._enc_backbone_str['predictor_num']
        self._objness, self._bbox2D_xy, self._bbox3D_dim, self._localZ = [], [], [], []
        self._sin, self._cos = [], []
        self._localZ_logvar, self._rad_logvar = [], []
        part_start = 0
        part_end = part_start
        for predIndex in range(pr_num):
            # objectness
            part_end += 1
            self._objness.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox2DXY_dim']
            self._bbox2D_xy.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox3D_dim']
            self._bbox3D_dim.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['localXYZ_dim']
            self._localZ.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['localXYZ_dim']
            self._localZ_logvar.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._sin.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._cos.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            self._rad_logvar.append(self._enc_output[..., part_start:part_end])
            part_start = part_end
            # print(part_end)
        self._objness = tf.sigmoid(tf.transpose(tf.stack(self._objness), [1, 2, 3, 0, 4]))
        self._bbox2D_xy = tf.sigmoid(tf.transpose(tf.stack(self._bbox2D_xy), [1, 2, 3, 0, 4]))
        self._bbox3D_dim = tf.transpose(tf.stack(self._bbox3D_dim), [1, 2, 3, 0, 4])
        self._bbox3D_dim = tf.exp(self._bbox3D_dim) * self._anchor_bbox3D
        self._localZ = tf.transpose(tf.stack(self._localZ), [1, 2, 3, 0, 4])
        if self._exp:
            self._localZ = tf.exp(self._localZ) * tf.expand_dims(self._anchor_z, axis=-1)
        else:
            self._localZ = self._localZ + tf.expand_dims(self._anchor_z, axis=-1)
        self._localZ_logvar = tf.clip_by_value(tf.transpose(tf.stack(self._localZ_logvar), [1, 2, 3, 0, 4]),
                                               clip_value_min=-10.0, clip_value_max=10.0)
        self._sin = tf.tanh(tf.transpose(tf.stack(self._sin), [1, 2, 3, 0, 4]))
        self._cos = tf.tanh(tf.transpose(tf.stack(self._cos), [1, 2, 3, 0, 4]))
        self._rad_logvar = tf.clip_by_value(tf.transpose(tf.stack(self._rad_logvar), [1, 2, 3, 0, 4]),
                                            clip_value_min=-10.0, clip_value_max=10.0)

    def _matmul3x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2]
        # return tf.stack([c0, c1, c2], axis=-1)
        c = tf.reduce_sum(a*tf.expand_dims(b, -2), axis=-1)
        return c

    def _matmul4x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2] + a[..., 0, 3] * b[..., 3]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2] + a[..., 1, 3] * b[..., 3]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2] + a[..., 2, 3] * b[..., 3]
        # c3 = a[..., 3, 0] * b[..., 0] + a[..., 3, 1] * b[..., 1] + a[..., 3, 2] * b[..., 2] + a[..., 3, 3] * b[..., 3]
        # return tf.stack([c0, c1, c2, c3], axis=-1)
        c = tf.reduce_sum(a*tf.expand_dims(b, -2), axis=-1)
        return c

    def _get3DBboxAnd2DPorj(self, projmat, R, t, lhw):
        # projmat : (batch, gridrow, girdcol, pred, 4x4)
        # R : (batch, gridrow, gridcol, pred, 3x3)
        # t : (batch, gridrow, gridcol, pred, 3)
        # lhw : (batch, gridrow, gridcol, pred, 3)
        dx, dy, dz = -lhw[..., 0] / 2., -lhw[..., 1] / 2., -lhw[..., 2] / 2.
        dxdydz = []
        for i in range(2):
            dy = -1. * dy
            for j in range(2):
                dx = -1. * dx
                for k in range(2):  # [x,y,z], [
                    dz = -1. * dz
                    dxdydz.append(tf.stack([dx, dy, dz], axis=-1))  # (8, b,gr,gc,pr,3)
        dxdydz = tf.transpose(tf.stack(dxdydz), [1, 2, 3, 4, 0, 5])  # (b,gr,gc,pr,8,3)
        R_tile = tf.transpose(tf.stack([R] * 8), [1, 2, 3, 4, 0, 5, 6])
        t_tile = tf.transpose(tf.stack([t] * 8), [1, 2, 3, 4, 0, 5])
        bbox3D8Points = self._matmul3x1(R_tile, dxdydz) + t_tile  # (b,gr,gc,pr,8,3)
        x_4d = tf.concat([bbox3D8Points, tf.expand_dims(tf.ones_like(dxdydz[..., 0]), axis=-1)], axis=-1)  # (b,gr,gc,pr,8,4)
        projmat_tile = tf.transpose(tf.stack([projmat] * 8), [1, 2, 3, 4, 0, 5, 6])
        bbox3D8PointsProj = self._matmul4x1(projmat_tile, x_4d)
        bbox3D8PointsProj = bbox3D8PointsProj[..., :2] / (tf.expand_dims(bbox3D8PointsProj[..., 2], axis=-1) + 1e-9)
        # print(bbox3D8PointsProj.shape)
        # select proj point
        x1 = tf.reduce_min(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size_org[..., 0]  # (b,gr,gc,pr)
        x2 = tf.reduce_max(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size_org[..., 0]
        y1 = tf.reduce_min(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size_org[..., 1]
        y2 = tf.reduce_max(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size_org[..., 1]
        # print(x1.shape)
        return bbox3D8Points, tf.stack([x1, y1, x2, y2], axis=-1)  # (b,gr,gc,pr,4)

    def _calcXYZ(self):
        len_grid_x, len_grid_y = tf.cast(tf.shape(self._offset_x)[2], tf.float32), tf.cast(tf.shape(self._offset_x)[1], tf.float32)
        # image_size : (row, col)
        objCenter2D_xz = (self._bbox2D_xy[..., 0] + self._offset_x) / len_grid_x * self._image_size_org[..., 0] * self._localZ[..., 0]
        objCenter2D_yz = (self._bbox2D_xy[..., 1] + self._offset_y) / len_grid_y * self._image_size_org[..., 1] * self._localZ[..., 0]
        objCenter2D_xyz = tf.stack([objCenter2D_xz, objCenter2D_yz, self._localZ[..., 0], tf.ones_like(self._localZ[..., 0])], axis=-1)
        self._localXYZ = self._matmul4x1(self._P2_inv_gt, objCenter2D_xyz)[..., 0:3]

    def _calcBbox3Dand2D(self):
        b, gr, gc, pr, _ = tf.shape(self._cos)
        zx_norm = tf.sqrt(tf.square(self._localXYZ[..., -1]) + tf.square(self._localXYZ[..., 0]))
        s_ray, c_ray = tf.expand_dims(self._localXYZ[..., 0] / zx_norm, axis=-1), tf.expand_dims(self._localXYZ[..., -1] / zx_norm, axis=-1)
        s_ray, c_ray = tf.constant(s_ray.numpy()), tf.constant(c_ray.numpy())
        self._sin_ry = s_ray * self._cos + c_ray * self._sin
        self._cos_ry = c_ray * self._cos - s_ray * self._sin
        # self._sin_ry, self._cos_ry = self._sin, self._cos
        zero = tf.zeros_like(self._cos_ry)
        one = tf.ones_like(self._cos_ry)
        self._R = tf.reshape(tf.concat([self._cos_ry, zero, self._sin_ry, zero, one, zero, -self._sin_ry, zero, self._cos_ry], axis=-1), [b, gr, gc, pr, 3, 3])
        self._bbox3D8Points, self._bbox2D_dim = self._get3DBboxAnd2DPorj(self._P2_gt, self._R, self._localXYZ, tf.constant(self._bbox3D_dim.numpy()))
        # self._bbox3D8Points
        # self._bbox2D_dim
        #
        self._ry = tf.atan2(self._sin_ry, self._cos_ry)
        self._alpha = tf.atan2(self._sin, self._cos)
        # self._bbox3D8Points
        # self._bbox2D_dim
        #

        #final output
        # self._localXYZ
        # self._R
        # self._bbox3D8Points
        # self._bbox2D_dim_normalized



