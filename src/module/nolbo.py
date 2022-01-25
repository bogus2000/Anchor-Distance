import src.net_core.darknet as darknet
import src.net_core.autoencoder3D as ae3D
import src.net_core.priornet as priornet
import numpy as np

from src.module.function import *
from src.box_IoU_rotation.box_intersection_2d import *


# @tf.RegisterGradient("DynamicPartition")
# def _DynamicPartitionGrads(op, *grads):
#     """Gradients for DynamicPartition."""
#     data = op.inputs[0]
#     indices = op.inputs[1]
#     num_partitions = op.get_attr("num_partitions")
#
#     prefix_shape = tf.shape(indices)
#     original_indices = tf.reshape(tf.range(tf.reduce_prod(prefix_shape)), prefix_shape)
#     partitioned_indices = tf.dynamic_partition(original_indices, indices, num_partitions)
#     reconstructed = tf.dynamic_stitch(partitioned_indices, grads)
#     reconstructed = tf.reshape(reconstructed, tf.shape(data))
#     return [reconstructed, None]

config = {
    'encoder_backbone':{
        'name' : 'nolbo_backbone',
        'predictor_num':9,
        'bbox2D_xy_dim':2, 'bbox3D_dim':3, 'orientation_dim':1,
        'localZ_dim':1,
        'inst_dim':10, 'z_inst_dim':16,
        'activation' : 'elu',
    },
    'encoder_head':{
        'name' : 'nolbo_head',
        'output_dim' : 5*(1+4+3+(2*3+3)+2*16),
        'filter_num_list':[1024,1024,1024],
        'filter_size_list':[3,3,3],
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
    'prior' : {
        'name' : 'priornet',
        'input_dim' : 10,  # class num (one-hot vector)
        'unit_num_list' : [64, 32, 16],
        'core_activation' : 'elu',
        'const_log_var' : 0.0,
    }
}

class nolbo(object):
    def __init__(self, nolbo_structure,
                 learning_rate=1e-4,
                 IoU2D_loss=True, IoU3D_loss=True,
                 solver='adam'):
        self._category_num = nolbo_structure['category_num']
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        # self._name = nolbo_structure['name']
        # self._predictor_num = nolbo_structure['predictor_num']
        # self._bbox2D_dim = nolbo_structure['bbox2D_dim']
        # self._bbox3D_dim = nolbo_structure['bbox3D_dim']
        # self._orientation_dim = nolbo_structure['orientation_dim']
        # self._inst_dim = nolbo_structure['inst_dim']
        # self._z_inst_dim = nolbo_structure['z_inst_dim']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._prior_str = nolbo_structure['prior']

        self._rad_var = (15.0/180.0 * 3.141593) ** 2

        self._IoU2D_loss, self._IoU3D_loss = IoU2D_loss, IoU3D_loss

        # # self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        self._buildModel()
        if solver == 'adam' or solver == 'Adam':
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif solver == 'sgd' or solver == 'SGD':
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=0.0005)

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
        self._priornet_car = priornet.priornet(structure=self._prior_str)
        self._classifier = darknet.classifier(name=None, input_shape=[64,], output_dim=self._category_num, activation='relu')
        print('done')

    def fit(self, inputs):
        self._getInputs(inputs=inputs)
        with tf.GradientTape() as tape:
            # get encoder output and loss
            self._enc_output = self._encoder_head(
                self._encoder_core_head(
                    self._encoder_core(self._input_images, training=True)
                    , training=True)
                , training=True)
            self._calcEncoderOutput()
            self._category_pred = self._classifier(self._z, training=True)

            self._getEncoderLoss()

            # # get (priornet, decoder) output and loss
            self._inst_mean_prior, self._inst_log_var_prior = self._priornet_car(self._carInstList, training=True)
            self._outputs = self._decoder(self._z_car, training=True)

            self._getDecoderAndPriorLoss()

            # get network parameter regulization loss
            # reg_loss = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses + self._decoder.losses + self._priornet.losses)
            # reg_loss = tf.reduce_sum(self._encoder_backbone.losses + self._encoder_head.losses)

            self._loss_objness = tf.reduce_mean(self._loss_objness, axis=0)
            # print(self._loss_objness.shape)
            self._loss_no_objness = tf.reduce_mean(self._loss_no_objness, axis=0)
            self._loss_bbox2D_hw = tf.reduce_mean(self._loss_bbox2D_hw, axis=0)
            self._loss_bbox2D_xy = tf.reduce_mean(self._loss_bbox2D_xy, axis=0)
            self._loss_bbox2D_CIOU = tf.reduce_mean(self._loss_bbox2D_CIOU, axis=0)
            self._loss_bbox3D = tf.reduce_mean(self._loss_bbox3D, axis=0)
            self._loss_bbox3D_IoU = tf.reduce_mean(self._loss_bbox3D_IoU, axis=0)
            self._loss_localXYZ = tf.reduce_mean(self._loss_localXYZ, axis=0)
            self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)
            self._loss_latents_kl = tf.reduce_mean(self._loss_latents_kl, axis=0)
            self._loss_prior_reg = tf.reduce_mean(self._loss_prior_reg, axis=0)
            self._loss_sincos = tf.reduce_mean(self._loss_sincos, axis=0)
            self._loss_sincos1 = tf.reduce_mean(self._loss_sincos1, axis=0)
            # self._loss_sincos_kl = tf.reduce_mean(self._loss_sincos_kl, axis=0)
            self._loss_category = tf.reduce_mean(self._loss_category, axis=0)

            # total loss
            total_loss = (
                    30.0 * self._loss_objness + 0.05 * self._loss_no_objness
                    + 20.0 * self._loss_bbox3D + 20.0 * self._loss_bbox2D_xy
                    # + 20.0 * self._loss_bbox3D_IoU + 20.0 * self._loss_bbox2D_CIOU
                    + 100.0 * self._loss_localXYZ
                    + self._loss_shape
                    + self._loss_latents_kl
                    + 0.01 * self._loss_prior_reg
                    + 100.0 * self._loss_sincos + 1000. * self._loss_sincos1
                    # + 0.01 * self._loss_sincos_kl
                    + 100.0 * self._loss_category
                    # + reg_loss
            )
            if self._IoU2D_loss:
                total_loss += 20.0 * (self._loss_bbox2D_CIOU + 0.1 * self._loss_bbox2D_hw)
            if self._IoU3D_loss:
                total_loss += self._loss_bbox3D_IoU

        trainable_variables = self._encoder_core.trainable_variables + self._encoder_core_head.trainable_variables + self._encoder_head.trainable_variables\
                                + self._decoder.trainable_variables + self._priornet_car.trainable_variables + self._classifier.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        # ==== evaluations
        self._objnessEval()
        self._obj_prb = tf.reduce_mean(self._obj_prb)
        self._no_obj_prb = tf.reduce_mean(self._no_obj_prb)

        TP, FP, FN = voxelPrecisionRecall(xTarget=self._output_images_gt, xPred=self._outputs)
        pr = tf.reduce_mean(TP / (TP + FP + 1e-10))
        rc = tf.reduce_mean(TP / (TP + FN + 1e-10))

        return self._loss_objness, self._loss_no_objness,\
               self._loss_bbox2D_CIOU, self._loss_bbox3D_IoU, \
               self._loss_bbox3D, self._loss_localXYZ, \
            self._loss_sincos, self._loss_sincos1,\
            self._loss_category, self._loss_shape, \
            self._obj_prb, self._no_obj_prb, \
            pr, rc

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_core.save_weights(os.path.join(save_path, file_name))

    def saveEncoderHead(self, save_path):
        file_name = self._enc_backbone_str['name'] + '_head'
        self._encoder_core_head.save_weights(os.path.join(save_path, file_name))
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))

    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)

    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))

    def savePriornet(self, save_path):
        file_name = self._prior_str['name']
        self._priornet_car.save_weights(os.path.join(save_path, file_name))

    def saveClassifier(self, save_path):
        file_name = 'classifier'
        self._classifier.save_weights(os.path.join(save_path, file_name))

    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriornet(save_path=save_path)
        self.saveClassifier(save_path=save_path)

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

    def loadPriornet(self, load_path):
        file_name = self._prior_str['name']
        self._priornet_car.load_weights(os.path.join(load_path, file_name))

    def loadClassifier(self, load_path):
        file_name = 'classifier'
        self._classifier.load_weights(os.path.join(load_path, file_name))

    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriornet(load_path=load_path)
        self.loadClassifier(load_path=load_path)

    def _getInputs(self, inputs):
        self._offset_x, self._offset_y, self._input_images,\
        self._objness_gt, self._objnessCar_gt,\
        self._bbox2D_dim_gt, self._bbox2D_xy_gt, self._bbox3D_dim_gt,\
        self._localXYZ_gt, self._rad_gt, \
        self._bbox3D8Points_gt,\
        self._image_size, self._P2_gt, self._P2_inv_gt, self._category_gt, \
        self._output_images_gt, self._carInstList, \
        self._anchor_z, self._anchor_bbox3D = inputs
        # self._output_images_gt, self._inst_vectors_gt, \

        self._offset_x = tf.convert_to_tensor(self._offset_x)
        self._offset_y = tf.convert_to_tensor(self._offset_y)
        self._input_images = tf.convert_to_tensor(self._input_images)
        self._objness_gt = tf.convert_to_tensor(self._objness_gt)
        self._objnessCar_gt = tf.convert_to_tensor(self._objnessCar_gt)
        self._bbox2D_dim_gt = tf.convert_to_tensor(self._bbox2D_dim_gt)
        self._bbox2D_xy_gt = tf.convert_to_tensor(self._bbox2D_xy_gt)
        self._bbox3D_dim_gt = tf.convert_to_tensor(self._bbox3D_dim_gt)
        self._localXYZ_gt = tf.convert_to_tensor(self._localXYZ_gt)
        self._rad_gt = tf.convert_to_tensor(self._rad_gt)
        self._bbox3D8Points_gt = tf.convert_to_tensor(self._bbox3D8Points_gt)
        # print(self._bbox3D8Points_gt.shape)
        self._image_size = tf.convert_to_tensor(self._image_size)
        self._P2_gt = tf.convert_to_tensor(self._P2_gt)
        self._P2_inv_gt = tf.convert_to_tensor(self._P2_inv_gt)
        self._category_gt = tf.convert_to_tensor(self._category_gt)
        self._output_images_gt = tf.convert_to_tensor(self._output_images_gt)
        self._carInstList = tf.convert_to_tensor(self._carInstList)
        self._anchor_z = tf.convert_to_tensor(self._anchor_z)
        self._anchor_bbox3D = tf.convert_to_tensor(self._anchor_bbox3D)

        self._sin_gt = tf.sin(self._rad_gt)
        self._cos_gt = tf.cos(self._rad_gt)

    def _calcEncoderOutput(self):
        self._encOutPartitioning()
        self._selectObjAndSampling()
        self._calcXYZ()
        self._calcBbox3Dand2D()
        self._getbbox2DIOU()

    def _getEncoderLoss(self):
        self._bbox2DLoss()
        self._objnessLoss()
        self._bbox2DLossCIOU()
        self._bbox3DLoss()
        self._bbox3DIoULoss()
        self._localXYZLoss()
        self._poseLoss()
        self._classificationLoss()

    def _getDecoderAndPriorLoss(self):
        self._objLatentAndShapeLoss()
        self._priorRegLoss()

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
        # self._rad_log_var = tf.transpose(tf.stack(self._rad_log_var), [1, 2, 3, 0, 4])
        # print(self._localZ.shape)
        # print(self._bbox3D_dim.shape)

    def _matmul3x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2]
        # return tf.stack([c0, c1, c2], axis=-1)
        c = tf.reduce_sum(a * tf.expand_dims(b, -2), axis=-1)
        return c

    def _matmul4x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2] + a[..., 0, 3] * b[..., 3]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2] + a[..., 1, 3] * b[..., 3]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2] + a[..., 2, 3] * b[..., 3]
        # c3 = a[..., 3, 0] * b[..., 0] + a[..., 3, 1] * b[..., 1] + a[..., 3, 2] * b[..., 2] + a[..., 3, 3] * b[..., 3]
        # return tf.stack([c0, c1, c2, c3], axis=-1)
        c = tf.reduce_sum(a * tf.expand_dims(b, -2), axis=-1)
        return c

    def _get3DBboxAnd2DPorj(self, projmat, R, t, lhw):
        # projmat : (batch, gridrow, girdcol, pred, 4x4)
        # R : (batch, gridrow, gridcol, pred, 3x3)
        # t : (batch, gridrow, gridcol, pred, 3)
        # lhw : (batch, gridrow, gridcol, pred, 3)
        dx, dy, dz = -lhw[...,0]/2., -lhw[...,1]/2., -lhw[...,2]/2.
        dxdydz = []
        for i in range(2):
            dy = -1. * dy
            for j in range(2):
                dx = -1. * dx
                for k in range(2): # [x,y,z], [
                    dz = -1. * dz
                    dxdydz.append(tf.stack([dx,dy,dz], axis=-1)) #(8, b,gr,gc,pr,3)
        dxdydz = tf.transpose(tf.stack(dxdydz), [1,2,3,4,0,5]) #(b,gr,gc,pr,8,3)
        R_tile = tf.transpose(tf.stack([R]*8), [1,2,3,4,0,5,6])
        t_tile = tf.transpose(tf.stack([t]*8), [1,2,3,4,0,5])
        bbox3D8Points = self._matmul3x1(R_tile, dxdydz) + t_tile #(b,gr,gc,pr,8,3)
        x_4d = tf.concat([bbox3D8Points, tf.expand_dims(tf.ones_like(dxdydz[...,0]),axis=-1)], axis=-1) #(b,gr,gc,pr,8,4)
        projmat_tile = tf.transpose(tf.stack([projmat]*8), [1,2,3,4,0,5,6])
        bbox3D8PointsProj = self._matmul4x1(projmat_tile, x_4d)
        bbox3D8PointsProj = bbox3D8PointsProj[..., :2] / (tf.expand_dims(bbox3D8PointsProj[..., 2], axis=-1) + 1e-9)
        # print(bbox3D8PointsProj.shape)
        # select proj point
        x1 = tf.reduce_min(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size[..., 0] # (b,gr,gc,pr)
        x2 = tf.reduce_max(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size[..., 0]
        y1 = tf.reduce_min(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size[..., 1]
        y2 = tf.reduce_max(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size[..., 1]
        # print(x1.shape)
        return bbox3D8Points, tf.stack([x1,y1,x2,y2], axis=-1) # (b,gr,gc,pr,4)

    def _calcXYZ(self):
        len_grid_x, len_grid_y = tf.cast(tf.shape(self._offset_x)[2], tf.float32), tf.cast(tf.shape(self._offset_x)[1], tf.float32)
        # image_size : (row, col)
        objCenter2D_xz = (self._bbox2D_xy[..., 0] + self._offset_x) / len_grid_x * self._image_size[..., 0] * self._localZ[..., 0]
        objCenter2D_yz = (self._bbox2D_xy[..., 1] + self._offset_y) / len_grid_y * self._image_size[..., 1] * self._localZ[..., 0]
        objCenter2D_xyz = tf.stack([objCenter2D_xz, objCenter2D_yz, self._localZ[...,0], tf.ones_like(self._localZ[...,0])], axis=-1)
        self._localXYZ = self._matmul4x1(self._P2_inv_gt, objCenter2D_xyz)[..., 0:3]

    def _calcBbox3Dand2D(self):
        b, gr, gc, pr, _ = tf.shape(self._cos)
        zx_norm = tf.sqrt(tf.square(self._localXYZ[..., -1]) + tf.square(self._localXYZ[..., 0]))
        s_ray, c_ray = tf.expand_dims(self._localXYZ[..., 0] / zx_norm, axis=-1), tf.expand_dims(self._localXYZ[..., -1] / zx_norm, axis=-1)
        s_ray, c_ray = tf.constant(s_ray.numpy()), tf.constant(c_ray.numpy())
        self._sin_ry = s_ray * self._cos + c_ray * self._sin
        self._cos_ry = c_ray * self._cos - s_ray * self._sin
        # self._cos_ry, self._sin_ry = self._cos, self._sin
        zero = tf.zeros_like(self._cos_ry)
        one = tf.ones_like(self._cos_ry)
        self._R = tf.reshape(tf.concat([self._cos_ry, zero, self._sin_ry,
                                        zero, one, zero,
                                        -self._sin_ry, zero, self._cos_ry]
                                       , axis=-1), [b, gr, gc, pr, 3, 3])
        self._bbox3D8Points, self._bbox2D_dim = self._get3DBboxAnd2DPorj(self._P2_gt, self._R, self._localXYZ, tf.constant(self._bbox3D_dim.numpy()))
        # self._bbox3D8Points
        # self._bbox2D_dim
        #

    def _getbbox2DIOU(self):
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = self._bbox2D_dim_gt[..., 0], self._bbox2D_dim_gt[..., 1], self._bbox2D_dim_gt[..., 2], self._bbox2D_dim_gt[..., 3]
        xmin, ymin, xmax, ymax = self._bbox2D_dim[..., 0], self._bbox2D_dim[..., 1], self._bbox2D_dim[..., 2], self._bbox2D_dim[..., 3]

        xmin_int = tf.math.maximum(xmin_gt, xmin)
        ymin_int = tf.math.maximum(ymin_gt, ymin)
        xmax_int = tf.math.minimum(xmax_gt, xmax)
        ymax_int = tf.math.minimum(ymax_gt, ymax)
        intersection_xlen = tf.maximum(xmax_int - xmin_int, 0.0)
        intersection_ylen = tf.maximum(ymax_int - ymin_int, 0.0)
        # print(intersection_xlen)
        # print(self._bbox2D_tile)
        intersection_area = intersection_xlen * intersection_ylen
        box_gt_area = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
        box_pr_area = (xmax - xmin) * (ymax - ymin)
        union_area = tf.maximum(box_gt_area + box_pr_area - intersection_area, 1e-9)
        self._IOU = tf.clip_by_value(intersection_area/union_area, 0., 1.)

        xmin_out = tf.minimum(xmin_gt, xmin)
        ymin_out = tf.minimum(ymin_gt, ymin)
        xmax_out = tf.maximum(xmax_gt, xmax)
        ymax_out = tf.maximum(ymax_gt, ymax)
        outer_xlen = tf.maximum(xmax_out - xmin_out, 0.)
        outer_ylen = tf.maximum(ymax_out - ymin_out, 0.)
        c2 = tf.square(outer_xlen) + tf.square(outer_ylen)  # sqr of diagonal length of max-outer box
        c2 = tf.maximum(c2, 1e-9)
        box_gt_x = (xmax_gt + xmin_gt) / 2.
        box_gt_y = (ymax_gt + ymin_gt) / 2.
        box_pr_x = (xmax + xmin) / 2.
        box_pr_y = (ymax + ymin) / 2.
        center_diff2 = tf.square(box_gt_x - box_pr_x) + tf.square(box_gt_y - box_pr_y)
        self._RDIOU = center_diff2 / c2

    def _bbox2DLoss(self):
        # tile shape = (batch, gridy, gridx, 2*predictornum, hwxy)
        square_d_xy = tf.reduce_sum(tf.square(self._bbox2D_xy - self._bbox2D_xy_gt), axis=-1)
        h_pred = self._bbox2D_dim[..., 3] - self._bbox2D_dim[..., 1]
        w_pred = self._bbox2D_dim[..., 2] - self._bbox2D_dim[..., 0]
        h_gt = self._bbox2D_dim_gt[..., 3] - self._bbox2D_dim_gt[..., 1]
        w_gt = self._bbox2D_dim_gt[..., 2] - self._bbox2D_dim_gt[..., 0]
        obj_mask = tf.reshape(self._objness_gt, tf.shape(self._objness_gt)[:-1])
        d_h = obj_mask * (h_pred - h_gt)
        d_w = obj_mask * (w_pred - w_gt)
        self._box_loss_scale = tf.constant((2. - w_gt * h_gt).numpy())
        xy_loss = obj_mask * self._box_loss_scale * square_d_xy
        hw_loss = obj_mask * self._box_loss_scale * (tf.square(d_h) + tf.square(d_w))
        self._loss_bbox2D_xy = tf.reduce_sum(xy_loss, axis=[1,2,3])
        self._loss_bbox2D_hw = tf.reduce_sum(hw_loss, axis=[1, 2, 3])


    def _objnessLoss(self):
        d_objness = -self._objness_gt * tf.math.log(self._objness + 1e-10) # * tf.square(tf.square(self._sin) + tf.square(self._cos))
        d_no_objness = - (1.0-self._objness_gt) * tf.math.log(1.0-self._objness + 1e-10)
        # d_no_objness = self._ignore_mask * d_no_objness[..., 0]
        d_no_objness = d_no_objness[..., 0]

        self._loss_objness = tf.reduce_sum(d_objness, axis=[1, 2, 3, 4])
        self._loss_no_objness = tf.reduce_sum(d_no_objness, axis=[1, 2, 3])

    def _smoothL1(self, x_src, x_trg, cond):
        # return tf.losses.huber(x_src, x_trg, cond)
        return tf.where(tf.abs(x_src - x_trg) > cond, tf.abs(x_src - x_trg) - 0.5 * cond, 0.5 / cond * tf.square(x_src - x_trg))

    def _bbox2DLossCIOU(self):
        obj_mask = self._objness_gt[..., 0]
        pi = 3.14159265358979323846
        # v = ((atan(w/h_gt) - atan(w/h_pr)) / (pi/2) )^2
        # bbox = hwxy
        h_pred = self._bbox2D_dim[..., 3] - self._bbox2D_dim[..., 1]
        w_pred = self._bbox2D_dim[..., 2] - self._bbox2D_dim[..., 0]
        h_gt = self._bbox2D_dim_gt[..., 3] - self._bbox2D_dim_gt[..., 1]
        w_gt = self._bbox2D_dim_gt[..., 2] - self._bbox2D_dim_gt[..., 0]
        ar_gt = w_gt / (h_gt + 1e-9)
        ar = w_pred / (h_pred + 1e-9)
        v = 4. / (pi * pi) * tf.square(tf.atan(ar_gt) - tf.atan(ar))
        alpha = v / (1. - self._IOU + v + 1e-9)
        loss_CIOU = obj_mask * (1. - self._IOU + self._RDIOU + alpha * v)
        # loss_CIOU = obj_mask * (1. - self._IOU)
        bbox_coor_loss = obj_mask * tf.reduce_sum(self._smoothL1(self._bbox2D_dim, self._bbox2D_dim_gt, 1e-4), axis=-1)
        # bbox_coor_loss = obj_mask * tf.reduce_sum(tf.square(self._bbox2D_dim - self._bbox2D_dim_gt), axis=-1)
        self._loss_bbox2D_CIOU = tf.reduce_sum(loss_CIOU + bbox_coor_loss, axis=[1, 2, 3])
        # loss_IOU = obj_mask * (1. - self._IOU)
        # self._loss_bbox2D_IOU = tf.reduce_sum(loss_IOU, axis=[1,2,3])

    def _bbox3DLoss(self):
        # self._loss_bbox3D = tf.reduce_sum(self._objness_gt * tf.square(self._bbox3D_dim_gt-self._bbox3D_dim), axis=[1,2,3,4])
        self._loss_bbox3D = tf.reduce_sum(self._objness_gt * self._smoothL1(self._bbox3D_dim_gt, self._bbox3D_dim, cond=1e-5), axis=[1,2,3,4])
        # self._loss_bbox3D = tf.reduce_sum(self._objness_gt * tf.expand_dims(self._box_loss_scale, axis=-1) * self._smoothL1(self._bbox3D_dim_gt, self._bbox3D_dim, 0.01), axis=[1, 2, 3, 4])

        # # obj_mask = tf.reshape(self._obj_mask, tf.shape(self._obj_mask)[:-1])
        # d = self._obj_mask * (self._bbox3D_tile - self._bbox3D_gt_tile)
        #
        # obj_mask = tf.reshape(self._obj_mask, tf.shape(self._obj_mask)[:-1])
        # box_loss_scale = obj_mask * (2. - self._bbox2D_gt_tile[..., 0] * self._bbox2D_gt_tile[..., 1])
        # d = box_loss_scale * tf.reduce_sum(d, axis=-1)
        #
        # # d shape = (batch, gridy, gridx, 2*predictornum, whl)
        # self._loss_bbox3D = tf.reduce_sum(tf.square(d), axis=[1, 2, 3])

    def _bbox3DIoULoss(self):
        IoU_3d = cal_iou_3d(box3d1=self._bbox3D8Points_gt, box3d2=self._bbox3D8Points,
                            lhw1=self._bbox3D_dim_gt, lhw2=self._bbox3D_dim)
        # print(IoU_3d.shape)
        obj_mask = self._objness_gt[..., 0]
        # print(obj_mask.shape)
        self._loss_bbox3D_IoU = obj_mask * (1. - IoU_3d)
        self._loss_bbox3D_IoU = tf.reduce_sum(self._loss_bbox3D_IoU , axis=[1, 2, 3])
        bbox_coor_loss = tf.reduce_sum(self._smoothL1(self._bbox3D8Points, self._bbox3D8Points_gt, 0.01), axis=[4, 5])
        # bbox_coor_loss = tf.reduce_sum(tf.square(self._bbox3D8Points - self._bbox3D8Points_gt), axis=[4, 5])
        # print(bbox_coor_loss.shape)
        bbox_coor_loss = tf.reduce_sum(obj_mask * bbox_coor_loss, axis=[1, 2, 3])
        # print(bbox_coor_loss.shape)
        self._loss_bbox3D_IoU += bbox_coor_loss
        # print(tf.reduce_sum(IoU_3d) / tf.reduce_sum(obj_mask))

    def _localXYZLoss(self):
        # loss_localXYZ_Bayesian = tf.square(self._localXYZ - self._localXYZ_gt) / (tf.exp(self._localXYZ_log_var) + 1e-9) + self._localXYZ_log_var
        # loss_localXYZ_Bayesian = self._localXYZ_log_var_tile
        # loss_localXYZ_Euclidian = tf.abs(self._localXYZ-self._localXYZ_gt)
        # loss_localXYZ_Euclidian = self._smoothL1(self._localXYZ, self._localXYZ_gt, 0.001) * tf.expand_dims(self._box_loss_scale, axis=-1)
        # loss_localXYZ_Euclidian = self._smoothL1(self._localXYZ, self._localXYZ_gt, 0.001)
        # loss_localXYZ_Euclidian = tf.square(self._localXYZ - self._localXYZ_gt)# * tf.expand_dims(self._box_loss_scale, axis=-1)
        loss_localXYZ_Euclidian = tf.expand_dims(tf.square(self._localZ[..., 0] - self._localXYZ_gt[..., 2]), axis=-1)# * tf.expand_dims(self._box_loss_scale, axis=-1)
        # loss_localXYZ = 0.1 * loss_localXYZ_Bayesian + 100.0 * loss_localXYZ_Euclidian
        # self._loss_localXYZ = tf.reduce_sum(self._objness_gt * 0.1 * loss_localXYZ_Bayesian, axis=[1, 2, 3, 4])
        self._loss_localXYZ = tf.reduce_sum((self._objness_gt * loss_localXYZ_Euclidian)[..., -1], axis=[1, 2, 3])

    def _getEV(self, sin, cos, radLogVar):
        Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
        Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
        Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
            -tf.exp(radLogVar)) * sin * sin
        Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
            -tf.exp(radLogVar)) * cos * cos
        logVarsin = tf.math.log(Varsin + 1e-7)
        logVarcos = tf.math.log(Varcos + 1e-7)
        return Esin, Ecos, logVarsin, logVarcos

    def _poseLoss(self):
        Esin, Ecos, logvarsin, logvarcos = self._getEV(sin=self._sin, cos=self._cos, radLogVar=tf.math.log(self._rad_var))
        Esin_gt, Ecos_gt, logvarsin_gt, logvarcos_gt = self._getEV(sin=self._sin_gt, cos=self._cos_gt, radLogVar=tf.math.log(self._rad_var))
        # loss_sin_kl = kl_loss(mean=Esin, logVar=logvarsin, mean_target=Esin_gt, logVar_target=logvarsin_gt)
        # loss_cos_kl = kl_loss(mean=Ecos, logVar=logvarcos, mean_target=Ecos_gt, logVar_target=logvarcos_gt)

        self._loss_sincos = tf.square(Esin-Esin_gt)/tf.exp(logvarsin_gt) + tf.square(Ecos-Ecos_gt)/tf.exp(logvarcos_gt)
        self._loss_sincos += tf.square(1. - (self._sin * self._sin_gt + self._cos * self._cos_gt))  # inner product
        self._loss_sincos += tf.square(self._sin - self._sin_gt) + tf.square(self._cos - self._cos_gt)  # cross product, 1st and 2nd component
        self._loss_sincos += tf.square(self._sin * self._cos_gt - self._cos * self._sin_gt)  # cross product, 3rd component

        scsquaresum = tf.square(self._sin) + tf.square(self._cos)
        self._loss_sincos1 = tf.square(1. - scsquaresum)

        # self._loss_sincos_bayesian = tf.reduce_sum(self._objness_gt * self._loss_sincos_bayesian, axis=[1, 2, 3, 4])
        # obj_mask = self._objness_gt[..., 0]
        # self._loss_sincos_kl = tf.reduce_sum(obj_mask * (loss_sin_kl + loss_cos_kl), axis=[1, 2, 3])
        self._loss_sincos = tf.reduce_sum(self._objness_gt * self._loss_sincos, axis=[1, 2, 3, 4])
        self._loss_sincos1 = tf.reduce_sum(self._objness_gt * self._loss_sincos1, axis=[1, 2, 3, 4])
                             # + 0.1 * tf.reduce_mean(self._loss_sincos1, axis=[1, 2, 3, 4])

    def _classificationLoss(self):
        self._category_pred = tf.nn.softmax(self._category_pred)
        self._loss_category = -tf.reduce_sum(self._category_gt * tf.math.log(self._category_pred + 1e-9), axis=-1)

    def _selectObjAndSampling(self):
        car_mask = tf.cast(self._objnessCar_gt[..., 0], tf.int32)  # (batch, row, col)
        self._latent_mean_car_sel = tf.dynamic_partition(self._latent_mean, car_mask, 2)[1]
        self._latent_log_var_car_sel = tf.dynamic_partition(self._latent_log_var, car_mask, 2)[1]
        self._z_car = sampling(mu=self._latent_mean_car_sel, logVar=self._latent_log_var_car_sel)
        obj_mask = tf.cast(self._objness_gt[..., 0], tf.int32)
        self._latent_mean_sel = tf.dynamic_partition(self._latent_mean, obj_mask, 2)[1]
        self._latent_log_var_sel = tf.dynamic_partition(self._latent_log_var, obj_mask, 2)[1]
        self._z = sampling(mu=self._latent_mean_sel, logVar=self._latent_log_var_sel)


    def _objLatentAndShapeLoss(self):
        self._loss_latents_kl = kl_loss(mean=self._latent_mean_car_sel, logVar=self._latent_log_var_car_sel,
                                        mean_target=self._inst_mean_prior, logVar_target=self._inst_log_var_prior)
        self._loss_shape = binary_loss(xPred=self._outputs, xTarget=self._output_images_gt, gamma=0.60, b_range=False)

    def _priorRegLoss(self):
        self._loss_prior_reg = regulizer_loss(z_mean=self._inst_mean_prior,
                                              z_logVar=self._inst_log_var_prior,
                                              dist_in_z_space=2.0 * self._enc_backbone_str['latent_dim'])

    def _objnessEval(self):
        self._obj_prb = (
                tf.reduce_sum(self._objness_gt * self._objness, axis=[1, 2, 3, 4])
                / tf.reduce_sum(self._objness_gt, axis=[1, 2, 3, 4]))

        self._no_obj_prb = (
                tf.reduce_sum((1.0 - self._objness_gt) * (1.0 - self._objness), axis=[1, 2, 3, 4])
                / tf.reduce_sum(1.0 - self._objness_gt, axis=[1, 2, 3, 4]))


class nolbo_bayesian(object):
    def __init__(self, nolbo_structure,
                 backbone_style=None, encoder_backbone=None,
                 learning_rate=1e-4,
                 IoU2D_loss=True, IoU3D_loss=True, exp=False,
                 solver='adam'):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        # self._name = nolbo_structure['name']
        # self._predictor_num = nolbo_structure['predictor_num']
        # self._bbox2D_dim = nolbo_structure['bbox2D_dim']
        # self._bbox3D_dim = nolbo_structure['bbox3D_dim']
        # self._orientation_dim = nolbo_structure['orientation_dim']
        # self._inst_dim = nolbo_structure['inst_dim']
        # self._z_inst_dim = nolbo_structure['z_inst_dim']
        self._enc_head_str = nolbo_structure['encoder_head']
        # self._dec_str = nolbo_structure['decoder']
        # self._prior_str = nolbo_structure['prior']

        self._rad_var = (15.0/180.0 * 3.141593) ** 2

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        self._IoU2D_loss, self._IoU3D_loss = IoU2D_loss, IoU3D_loss
        self._exp = exp

        # # self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        self._buildModel()
        if solver == 'adam' or solver == 'Adam':
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif solver == 'sgd' or solver == 'SGD':
            self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=0.0005)

    def _buildModel(self):
        print('build Models...')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name=self._enc_backbone_str['name'])
        #==============set encoder head
        self._encoder_head = darknet.head2D(name=self._enc_head_str['name'],
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=self._enc_head_str['output_dim'],
                                            filter_num_list=self._enc_head_str['filter_num_list'],
                                            filter_size_list=self._enc_head_str['filter_size_list'],
                                            last_pooling=None, activation=self._enc_head_str['activation'])
        # #==============set decoder3D
        # self._decoder = ae3D.decoder3D(structure=self._dec_str)
        # self._priornet = priornet.priornet(structure=self._prior_str)
        print('done')

    def fit(self, inputs):
        self._getInputs(inputs=inputs)
        with tf.GradientTape() as tape:
            # get encoder output and loss
            self._input_images = self._input_images / 255.
            self._enc_output = self._encoder_backbone(self._input_images, training=True)
            self._enc_output = self._encoder_head(self._enc_output, training=True)
            self._calcEncoderOutput()

            # # get (priornet, decoder) output and loss
            # self._inst_mean_prior, self._inst_log_var_prior = self._priornet(self._inst_vectors_gt, training=True)
            # self._selectObjFromTile()
            # self._latents = sampling(mu=self._inst_mean_sel, logVar=self._inst_log_var_sel)
            # self._outputs = self._decoder(self._latents, training=True)

            self._getEncoderLoss()
            # self._getDecoderAndPriorLoss()

            # get network parameter regulization loss
            # reg_loss = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses + self._decoder.losses + self._priornet.losses)
            reg_loss = tf.reduce_sum(self._encoder_backbone.losses + self._encoder_head.losses)

            self._loss_objness = tf.reduce_mean(self._loss_objness, axis=0)
            # print(self._loss_objness.shape)
            self._loss_no_objness = tf.reduce_mean(self._loss_no_objness, axis=0)
            self._loss_bbox2D_hw = tf.reduce_mean(self._loss_bbox2D_hw, axis=0)
            self._loss_bbox2D_xy = tf.reduce_mean(self._loss_bbox2D_xy, axis=0)
            self._loss_bbox2D_CIOU = tf.reduce_mean(self._loss_bbox2D_CIOU, axis=0)
            self._loss_bbox3D = tf.reduce_mean(self._loss_bbox3D, axis=0)
            self._loss_bbox3D_IoU = tf.reduce_mean(self._loss_bbox3D_IoU, axis=0)
            self._loss_localXYZ_Bayesian = tf.reduce_mean(self._loss_localXYZ_Bayesian, axis=0)
            self._loss_localXYZ_Euclidian = tf.reduce_mean(self._loss_localXYZ_Euclidian, axis=0)
            # self._loss_shape = tf.reduce_mean(self._loss_shape, axis=0)
            # self._loss_latents_kl = tf.reduce_mean(self._loss_latents_kl, axis=0)
            # self._loss_prior_reg = tf.reduce_mean(self._loss_prior_reg, axis=0)
            self._loss_sincos_bayesian = tf.reduce_mean(self._loss_sincos_bayesian, axis=0)
            self._loss_sincos = tf.reduce_mean(self._loss_sincos, axis=0)
            self._loss_sincos1 = tf.reduce_mean(self._loss_sincos1, axis=0)

            # total loss
            total_loss = (
                    50.0 * self._loss_objness + 0.01 * self._loss_no_objness
                    + 20.0 * self._loss_bbox3D + 20.0 * self._loss_bbox2D_xy
                    + 0.001 * self._loss_localXYZ_Bayesian + self._loss_localXYZ_Euclidian
                    # + self._loss_latents_kl
                    + 100.0 * self._loss_sincos + 1000. * self._loss_sincos1
                    + 0.001 * self._loss_sincos_bayesian
                    # + reg_loss
            )
            if self._IoU2D_loss:
                total_loss += 20.0 * (self._loss_bbox2D_CIOU + 0.1 * self._loss_bbox2D_hw)
            if self._IoU3D_loss:
                total_loss += self._loss_bbox3D_IoU

        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        # ==== evaluations
        self._objnessEval()
        self._obj_prb = tf.reduce_mean(self._obj_prb)
        self._no_obj_prb = tf.reduce_mean(self._no_obj_prb)

        # TP, FP, FN = voxelPrecisionRecall(xTarget=self._output_images_gt, xPred=self._outputs)
        # pr = tf.reduce_mean(TP / (TP + FP + 1e-10))
        # rc = tf.reduce_mean(TP / (TP + FN + 1e-10))

        return self._loss_objness, self._loss_no_objness,\
               self._loss_bbox2D_CIOU, self._loss_bbox3D_IoU, \
               self._loss_bbox3D, self._loss_localXYZ_Euclidian, self._loss_localXYZ_Bayesian, \
            self._loss_sincos, self._loss_sincos1, self._loss_sincos_bayesian,\
            self._obj_prb, self._no_obj_prb

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))
    def saveEncoderHead(self, save_path):
        file_name = self._enc_head_str['name']
        self._encoder_head.save_weights(os.path.join(save_path, file_name))
    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)
    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)

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

    def _getInputs(self, inputs):
        self._offset_x, self._offset_y,\
        self._input_images,\
        self._objness_gt, self._bbox2D_dim_gt, self._bbox2D_xy_gt, self._bbox3D_dim_gt,\
        self._localXYZ_gt,\
        self._rad_gt, \
        self._bbox3D8Points_gt,\
        self._image_size, \
        self._P2_gt, self._P2_inv_gt,\
        self._anchor_z, self._anchor_bbox3D = inputs
        # self._output_images_gt, self._inst_vectors_gt, \

        self._offset_x = tf.convert_to_tensor(self._offset_x)
        self._offset_y = tf.convert_to_tensor(self._offset_y)
        self._input_images = tf.convert_to_tensor(self._input_images)
        self._objness_gt = tf.convert_to_tensor(self._objness_gt)
        self._bbox2D_dim_gt = tf.convert_to_tensor(self._bbox2D_dim_gt)
        self._bbox2D_xy_gt = tf.convert_to_tensor(self._bbox2D_xy_gt)
        self._bbox3D_dim_gt = tf.convert_to_tensor(self._bbox3D_dim_gt)
        self._localXYZ_gt = tf.convert_to_tensor(self._localXYZ_gt)
        self._rad_gt = tf.convert_to_tensor(self._rad_gt)
        self._bbox3D8Points_gt = tf.convert_to_tensor(self._bbox3D8Points_gt)
        # print(self._bbox3D8Points_gt.shape)
        self._image_size = tf.convert_to_tensor(self._image_size)
        self._P2_gt = tf.convert_to_tensor(self._P2_gt)
        self._P2_inv_gt = tf.convert_to_tensor(self._P2_inv_gt)
        # self._output_images_gt = tf.convert_to_tensor(self._output_images_gt)
        # self._inst_vectors_gt = tf.convert_to_tensor(self._inst_vectors_gt)
        self._anchor_z = tf.convert_to_tensor(self._anchor_z)
        self._anchor_bbox3D = tf.convert_to_tensor(self._anchor_bbox3D)

        self._sin_gt = tf.sin(self._rad_gt)
        self._cos_gt = tf.cos(self._rad_gt)

    def _calcEncoderOutput(self):
        self._encOutPartitioning()
        self._calcXYZ()
        self._calcBbox3Dand2D()
        # self._createTiles()
        self._getbbox2DIOU()
        # self._getObjMaskAndObjGT()

    def _getEncoderLoss(self):
        self._bbox2DLoss()
        self._objnessLoss()
        self._bbox2DLossCIOU()
        self._bbox3DLoss()
        self._bbox3DIoULoss()
        self._localXYZLoss()
        self._poseLoss()

    # def _getDecoderAndPriorLoss(self):
    #     self._objLatentAndShapeLoss()
    #     self._priorRegLoss()

    def _encOutPartitioning(self):
        pr_num = self._enc_backbone_str['predictor_num']
        self._objness, self._bbox2D_xy, self._bbox3D_dim = [], [], []
        self._localZ, self._localZ_logvar = [], []
        self._sin, self._cos, self._rad_logvar = [], [], []
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
        self._bbox3D_dim = tf.transpose(tf.stack(self._bbox3D_dim), [1,2,3,0,4])
        self._bbox3D_dim = tf.exp(self._bbox3D_dim) * self._anchor_bbox3D
        # print(self._bbox3D_dim.shape)
        self._localZ = tf.transpose(tf.stack(self._localZ), [1, 2, 3, 0, 4])
        if self._exp:
            self._localZ = tf.exp(self._localZ) * tf.expand_dims(self._anchor_z, axis=-1)
        else:
            self._localZ = self._localZ + tf.expand_dims(self._anchor_z, axis=-1)
        self._localZ_logvar = tf.clip_by_value(tf.transpose(tf.stack(self._localZ_logvar), [1, 2, 3, 0, 4]), clip_value_min=-2.0, clip_value_max=2.0)
        self._sin = tf.tanh(tf.transpose(tf.stack(self._sin), [1, 2, 3, 0, 4]))
        self._cos = tf.tanh(tf.transpose(tf.stack(self._cos), [1, 2, 3, 0, 4]))
        self._rad_logvar = tf.clip_by_value(tf.transpose(tf.stack(self._rad_logvar), [1, 2, 3, 0, 4]), clip_value_min=-1.0, clip_value_max=1.0)

    def _matmul3x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2]
        # return tf.stack([c0, c1, c2], axis=-1)
        c = tf.reduce_sum(a * tf.expand_dims(b, -2), axis=-1)
        return c

    def _matmul4x1(self, a, b):
        # c0 = a[..., 0, 0] * b[..., 0] + a[..., 0, 1] * b[..., 1] + a[..., 0, 2] * b[..., 2] + a[..., 0, 3] * b[..., 3]
        # c1 = a[..., 1, 0] * b[..., 0] + a[..., 1, 1] * b[..., 1] + a[..., 1, 2] * b[..., 2] + a[..., 1, 3] * b[..., 3]
        # c2 = a[..., 2, 0] * b[..., 0] + a[..., 2, 1] * b[..., 1] + a[..., 2, 2] * b[..., 2] + a[..., 2, 3] * b[..., 3]
        # c3 = a[..., 3, 0] * b[..., 0] + a[..., 3, 1] * b[..., 1] + a[..., 3, 2] * b[..., 2] + a[..., 3, 3] * b[..., 3]
        # return tf.stack([c0, c1, c2, c3], axis=-1)
        c = tf.reduce_sum(a * tf.expand_dims(b, -2), axis=-1)
        return c

    def _get3DBboxAnd2DPorj(self, projmat, R, t, lhw):
        # projmat : (batch, gridrow, girdcol, pred, 4x4)
        # R : (batch, gridrow, gridcol, pred, 3x3)
        # t : (batch, gridrow, gridcol, pred, 3)
        # lhw : (batch, gridrow, gridcol, pred, 3)
        dx, dy, dz = -lhw[...,0]/2., -lhw[...,1]/2., -lhw[...,2]/2.
        dxdydz = []
        for i in range(2):
            dy = -1. * dy
            for j in range(2):
                dx = -1. * dx
                for k in range(2): # [x,y,z], [
                    dz = -1. * dz
                    dxdydz.append(tf.stack([dx,dy,dz], axis=-1)) #(8, b,gr,gc,pr,3)
        dxdydz = tf.transpose(tf.stack(dxdydz), [1,2,3,4,0,5]) #(b,gr,gc,pr,8,3)
        R_tile = tf.transpose(tf.stack([R]*8), [1,2,3,4,0,5,6])
        t_tile = tf.transpose(tf.stack([t]*8), [1,2,3,4,0,5])
        bbox3D8Points = self._matmul3x1(R_tile, dxdydz) + t_tile #(b,gr,gc,pr,8,3)
        x_4d = tf.concat([bbox3D8Points, tf.expand_dims(tf.ones_like(dxdydz[...,0]),axis=-1)], axis=-1) #(b,gr,gc,pr,8,4)
        projmat_tile = tf.transpose(tf.stack([projmat]*8), [1,2,3,4,0,5,6])
        bbox3D8PointsProj = self._matmul4x1(projmat_tile, x_4d)
        bbox3D8PointsProj = bbox3D8PointsProj[..., :2] / (tf.expand_dims(bbox3D8PointsProj[..., 2], axis=-1) + 1e-9)
        # print(bbox3D8PointsProj.shape)
        # select proj point
        x1 = tf.reduce_min(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size[..., 0] # (b,gr,gc,pr)
        x2 = tf.reduce_max(bbox3D8PointsProj[..., 0], axis=-1) / self._image_size[..., 0]
        y1 = tf.reduce_min(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size[..., 1]
        y2 = tf.reduce_max(bbox3D8PointsProj[..., 1], axis=-1) / self._image_size[..., 1]
        # print(x1.shape)
        return bbox3D8Points, tf.stack([x1,y1,x2,y2], axis=-1) # (b,gr,gc,pr,4)

    def _calcXYZ(self):
        len_grid_x, len_grid_y = tf.cast(tf.shape(self._offset_x)[2], tf.float32), tf.cast(tf.shape(self._offset_x)[1], tf.float32)
        # image_size : (row, col)
        objCenter2D_xz = (self._bbox2D_xy[..., 0] + self._offset_x) / len_grid_x * self._image_size[..., 0] * self._localZ[..., 0]
        objCenter2D_yz = (self._bbox2D_xy[..., 1] + self._offset_y) / len_grid_y * self._image_size[..., 1] * self._localZ[..., 0]
        objCenter2D_xyz = tf.stack([objCenter2D_xz, objCenter2D_yz, self._localZ[...,0], tf.ones_like(self._localZ[...,0])], axis=-1)
        self._localXYZ = self._matmul4x1(self._P2_inv_gt, objCenter2D_xyz)[..., 0:3]

    def _calcBbox3Dand2D(self):
        b, gr, gc, pr, _ = tf.shape(self._cos)
        zx_norm = tf.sqrt(tf.square(self._localXYZ[..., -1]) + tf.square(self._localXYZ[..., 0]))
        s_ray, c_ray = tf.expand_dims(self._localXYZ[..., 0] / zx_norm, axis=-1), tf.expand_dims(self._localXYZ[..., -1] / zx_norm, axis=-1)
        s_ray, c_ray = tf.constant(s_ray.numpy()), tf.constant(c_ray.numpy())
        self._sin_ry = s_ray * self._cos + c_ray * self._sin
        self._cos_ry = c_ray * self._cos - s_ray * self._sin
        # self._cos_ry, self._sin_ry = self._cos, self._sin
        zero = tf.zeros_like(self._cos_ry)
        one = tf.ones_like(self._cos_ry)
        self._R = tf.reshape(tf.concat([self._cos_ry, zero, self._sin_ry,
                                        zero, one, zero,
                                        -self._sin_ry, zero, self._cos_ry]
                                       , axis=-1), [b, gr, gc, pr, 3, 3])
        self._bbox3D8Points, self._bbox2D_dim = self._get3DBboxAnd2DPorj(self._P2_gt, self._R, self._localXYZ, tf.constant(self._bbox3D_dim.numpy()))
        # self._bbox3D8Points
        # self._bbox2D_dim
        #

    def _getbbox2DIOU(self):
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = self._bbox2D_dim_gt[..., 0], self._bbox2D_dim_gt[..., 1], self._bbox2D_dim_gt[..., 2], self._bbox2D_dim_gt[..., 3]
        xmin, ymin, xmax, ymax = self._bbox2D_dim[..., 0], self._bbox2D_dim[..., 1], self._bbox2D_dim[..., 2], self._bbox2D_dim[..., 3]

        xmin_int = tf.math.maximum(xmin_gt, xmin)
        ymin_int = tf.math.maximum(ymin_gt, ymin)
        xmax_int = tf.math.minimum(xmax_gt, xmax)
        ymax_int = tf.math.minimum(ymax_gt, ymax)
        intersection_xlen = tf.maximum(xmax_int - xmin_int, 0.0)
        intersection_ylen = tf.maximum(ymax_int - ymin_int, 0.0)
        # print(intersection_xlen)
        # print(self._bbox2D_tile)
        intersection_area = intersection_xlen * intersection_ylen
        box_gt_area = (xmax_gt - xmin_gt) * (ymax_gt - ymin_gt)
        box_pr_area = (xmax - xmin) * (ymax - ymin)
        union_area = tf.maximum(box_gt_area + box_pr_area - intersection_area, 1e-9)
        self._IOU = tf.clip_by_value(intersection_area/union_area, 0., 1.)

        xmin_out = tf.minimum(xmin_gt, xmin)
        ymin_out = tf.minimum(ymin_gt, ymin)
        xmax_out = tf.maximum(xmax_gt, xmax)
        ymax_out = tf.maximum(ymax_gt, ymax)
        outer_xlen = tf.maximum(xmax_out - xmin_out, 0.)
        outer_ylen = tf.maximum(ymax_out - ymin_out, 0.)
        c2 = tf.square(outer_xlen) + tf.square(outer_ylen)  # sqr of diagonal length of max-outer box
        c2 = tf.maximum(c2, 1e-9)
        box_gt_x = (xmax_gt + xmin_gt) / 2.
        box_gt_y = (ymax_gt + ymin_gt) / 2.
        box_pr_x = (xmax + xmin) / 2.
        box_pr_y = (ymax + ymin) / 2.
        center_diff2 = tf.square(box_gt_x - box_pr_x) + tf.square(box_gt_y - box_pr_y)
        self._RDIOU = center_diff2 / c2

    def _bbox2DLoss(self):
        # tile shape = (batch, gridy, gridx, 2*predictornum, hwxy)
        square_d_xy = tf.reduce_sum(tf.square(self._bbox2D_xy - self._bbox2D_xy_gt), axis=-1)
        h_pred = self._bbox2D_dim[..., 3] - self._bbox2D_dim[..., 1]
        w_pred = self._bbox2D_dim[..., 2] - self._bbox2D_dim[..., 0]
        h_gt = self._bbox2D_dim_gt[..., 3] - self._bbox2D_dim_gt[..., 1]
        w_gt = self._bbox2D_dim_gt[..., 2] - self._bbox2D_dim_gt[..., 0]
        obj_mask = tf.reshape(self._objness_gt, tf.shape(self._objness_gt)[:-1])
        d_h = obj_mask * (h_pred - h_gt)
        d_w = obj_mask * (w_pred - w_gt)
        self._box_loss_scale = tf.constant((2. - w_gt * h_gt).numpy())
        xy_loss = obj_mask * self._box_loss_scale * square_d_xy
        hw_loss = obj_mask * self._box_loss_scale * (tf.square(d_h) + tf.square(d_w))
        self._loss_bbox2D_xy = tf.reduce_sum(xy_loss, axis=[1,2,3])
        self._loss_bbox2D_hw = tf.reduce_sum(hw_loss, axis=[1, 2, 3])


    def _objnessLoss(self):
        d_objness = -self._objness_gt * tf.math.log(self._objness + 1e-10) # * tf.square(tf.square(self._sin) + tf.square(self._cos))
        d_no_objness = - (1.-self._objness_gt) * tf.math.log(1.-self._objness + 1e-10)
        # d_no_objness = self._ignore_mask * d_no_objness[..., 0]
        # d_no_objness = d_no_objness[..., 0]

        self._loss_objness = tf.reduce_sum(d_objness, axis=[1, 2, 3, 4])
        self._loss_no_objness = tf.reduce_sum(d_no_objness, axis=[1, 2, 3, 4])

    def _smoothL1(self, x_src, x_trg, cond):
        # return tf.losses.huber(x_src, x_trg, cond)
        return tf.where(tf.abs(x_src - x_trg) > cond, tf.abs(x_src - x_trg) - 0.5 * cond, 0.5 / cond * tf.square(x_src - x_trg))

    def _bbox2DLossCIOU(self):
        obj_mask = self._objness_gt[..., 0]
        pi = 3.14159265358979323846
        # v = ((atan(w/h_gt) - atan(w/h_pr)) / (pi/2) )^2
        # bbox = hwxy
        h_pred = self._bbox2D_dim[..., 3] - self._bbox2D_dim[..., 1]
        w_pred = self._bbox2D_dim[..., 2] - self._bbox2D_dim[..., 0]
        h_gt = self._bbox2D_dim_gt[..., 3] - self._bbox2D_dim_gt[..., 1]
        w_gt = self._bbox2D_dim_gt[..., 2] - self._bbox2D_dim_gt[..., 0]
        ar_gt = w_gt / (h_gt + 1e-9)
        ar = w_pred / (h_pred + 1e-9)
        v = 4. / (pi * pi) * tf.square(tf.atan(ar_gt) - tf.atan(ar))
        alpha = v / (1. - self._IOU + v + 1e-9)
        loss_CIOU = obj_mask * (1. - self._IOU + self._RDIOU + alpha * v)
        # loss_CIOU = obj_mask * (1. - self._IOU)
        bbox_coor_loss = obj_mask * tf.reduce_sum(self._smoothL1(self._bbox2D_dim, self._bbox2D_dim_gt, 1e-4), axis=-1)
        # bbox_coor_loss = obj_mask * tf.reduce_sum(tf.square(self._bbox2D_dim - self._bbox2D_dim_gt), axis=-1)
        self._loss_bbox2D_CIOU = tf.reduce_sum(loss_CIOU + bbox_coor_loss, axis=[1, 2, 3])
        # loss_IOU = obj_mask * (1. - self._IOU)
        # self._loss_bbox2D_IOU = tf.reduce_sum(loss_IOU, axis=[1,2,3])

    def _bbox3DLoss(self):
        self._loss_bbox3D = tf.reduce_sum(self._objness_gt * tf.square(self._bbox3D_dim_gt-self._bbox3D_dim), axis=[1,2,3,4])
        # self._loss_bbox3D = tf.reduce_sum(self._objness_gt * self._smoothL1(self._bbox3D_dim_gt, self._bbox3D_dim, cond=1e-5), axis=[1,2,3,4])
        # self._loss_bbox3D = tf.reduce_sum(self._objness_gt * tf.expand_dims(self._box_loss_scale, axis=-1) * self._smoothL1(self._bbox3D_dim_gt, self._bbox3D_dim, 0.01), axis=[1, 2, 3, 4])

        # # obj_mask = tf.reshape(self._obj_mask, tf.shape(self._obj_mask)[:-1])
        # d = self._obj_mask * (self._bbox3D_tile - self._bbox3D_gt_tile)
        #
        # obj_mask = tf.reshape(self._obj_mask, tf.shape(self._obj_mask)[:-1])
        # box_loss_scale = obj_mask * (2. - self._bbox2D_gt_tile[..., 0] * self._bbox2D_gt_tile[..., 1])
        # d = box_loss_scale * tf.reduce_sum(d, axis=-1)
        #
        # # d shape = (batch, gridy, gridx, 2*predictornum, whl)
        # self._loss_bbox3D = tf.reduce_sum(tf.square(d), axis=[1, 2, 3])

    def _bbox3DIoULoss(self):
        IoU_3d = cal_iou_3d(box3d1=self._bbox3D8Points_gt, box3d2=self._bbox3D8Points,
                            lhw1=self._bbox3D_dim_gt, lhw2=self._bbox3D_dim)
        # print(IoU_3d.shape)
        obj_mask = self._objness_gt[..., 0]
        # print(obj_mask.shape)
        self._loss_bbox3D_IoU = obj_mask * (1. - IoU_3d)
        self._loss_bbox3D_IoU = tf.reduce_sum(self._loss_bbox3D_IoU , axis=[1, 2, 3])
        bbox_coor_loss = tf.reduce_sum(self._smoothL1(self._bbox3D8Points, self._bbox3D8Points_gt, 0.01), axis=[4, 5])
        # bbox_coor_loss = tf.reduce_sum(tf.square(self._bbox3D8Points - self._bbox3D8Points_gt), axis=[4, 5])
        # print(bbox_coor_loss.shape)
        bbox_coor_loss = tf.reduce_sum(obj_mask * bbox_coor_loss, axis=[1, 2, 3])
        # print(bbox_coor_loss.shape)
        self._loss_bbox3D_IoU += bbox_coor_loss
        # print(tf.reduce_sum(IoU_3d) / tf.reduce_sum(obj_mask))

    def _localXYZLoss(self):
        localZ_gt = tf.expand_dims(self._localXYZ_gt[..., -1], axis=-1)
        loss_localXYZ_Bayesian = tf.square(self._localZ - localZ_gt) / (tf.exp(self._localZ_logvar) + 1e-9) + self._localZ_logvar
        # loss_localXYZ_Bayesian = self._localXYZ_log_var_tile
        # loss_localXYZ_Euclidian = tf.abs(self._localXYZ-self._localXYZ_gt)
        # loss_localXYZ_Euclidian = self._smoothL1(self._localXYZ, self._localXYZ_gt, 0.001) * tf.expand_dims(self._box_loss_scale, axis=-1)
        # loss_localXYZ_Euclidian = self._smoothL1(self._localXYZ[..., -1], self._localXYZ_gt[..., -1], 0.001) * self._box_loss_scale
        loss_localXYZ_Euclidian = tf.square(self._localZ - localZ_gt)# * tf.expand_dims(self._box_loss_scale, axis=-1)
        # loss_localXYZ_Euclidian = 100. * tf.expand_dims(tf.square(self._localZ[..., 0] - self._localXYZ_gt[..., 2]), axis=-1)# * tf.expand_dims(self._box_loss_scale, axis=-1)
        # self._loss_localXYZ = tf.reduce_sum(self._objness_gt * 0.1 * loss_localXYZ_Bayesian, axis=[1, 2, 3, 4])
        self._loss_localXYZ_Bayesian = 100. * tf.reduce_sum(self._objness_gt * loss_localXYZ_Bayesian, axis=[1, 2, 3, 4])
        self._loss_localXYZ_Euclidian = 100. * tf.reduce_sum(self._objness_gt * loss_localXYZ_Euclidian, axis=[1, 2, 3, 4])

    def _getEV(self, sin, cos, radLogVar):
        Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
        Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
        Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
            -tf.exp(radLogVar)) * sin * sin
        Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
            -tf.exp(radLogVar)) * cos * cos
        logVarsin = tf.math.log(Varsin + 1e-7)
        logVarcos = tf.math.log(Varcos + 1e-7)
        return Esin, Ecos, logVarsin, logVarcos

    def _poseLoss(self):
        Esin, Ecos, _, _ = self._getEV(sin=self._sin, cos=self._cos, radLogVar=tf.math.log(self._rad_var))
        Esin_gt, Ecos_gt, logvarsin_gt, logvarcos_gt = self._getEV(sin=self._sin_gt, cos=self._cos_gt, radLogVar=tf.math.log(self._rad_var))
        #
        _, _, logvarsin, logvarcos = self._getEV(sin=self._sin, cos=self._cos, radLogVar=self._rad_logvar)
        self._loss_sincos_bayesian = tf.square(self._sin - self._sin_gt)/tf.exp(logvarsin) + logvarsin
        self._loss_sincos_bayesian += tf.square(self._cos - self._cos_gt)/tf.exp(logvarcos) + logvarcos
        #
        self._loss_sincos = tf.square(Esin-Esin_gt)/tf.exp(logvarsin_gt) + tf.square(Ecos-Ecos_gt)/tf.exp(logvarcos_gt)
        self._loss_sincos += tf.square(1. - (self._sin * self._sin_gt + self._cos * self._cos_gt))  # inner product
        self._loss_sincos += tf.square(self._sin - self._sin_gt) + tf.square(self._cos - self._cos_gt)  # cross product, 1st and 2nd component
        self._loss_sincos += tf.square(self._sin * self._cos_gt - self._cos * self._sin_gt)  # cross product, 3rd component

        scsquaresum = tf.square(self._sin) + tf.square(self._cos)
        self._loss_sincos1 = tf.square(1. - scsquaresum)

        self._loss_sincos_bayesian = tf.reduce_sum(self._objness_gt * self._loss_sincos_bayesian, axis=[1, 2, 3, 4])
        self._loss_sincos = tf.reduce_sum(self._objness_gt * self._loss_sincos, axis=[1, 2, 3, 4])
        self._loss_sincos1 = tf.reduce_sum(self._objness_gt * self._loss_sincos1, axis=[1, 2, 3, 4])
                             # + 0.1 * tf.reduce_mean(self._loss_sincos1, axis=[1, 2, 3, 4])

    def _objnessEval(self):
        self._obj_prb = (
                tf.reduce_sum(self._objness_gt * self._objness, axis=[1, 2, 3, 4])
                / tf.reduce_sum(self._objness_gt, axis=[1, 2, 3, 4]))

        self._no_obj_prb = (
                tf.reduce_sum((1.0 - self._objness_gt) * (1.0 - self._objness), axis=[1, 2, 3, 4])
                / tf.reduce_sum(1.0 - self._objness_gt, axis=[1, 2, 3, 4]))


class nolbo_single(object):
    def __init__(self, encoder_backbone=None,
                 decoder_structure=None,
                 prior_class_structure=None,
                 prior_inst_structure=None,
                 BATCH_SIZE_PER_REPLICA=32, strategy=None,
                 learning_rate = 1e-4
                 ):
        self._rad_var = (15.0/180.0 * 3.141593) ** 2
        self._dec_str = decoder_structure
        self._prior_cl_str = prior_class_structure
        self._prior_inst_str = prior_inst_structure

        self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        self._GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        with self._strategy.scope():
            self._encoder_backbone = encoder_backbone
            self._buildModel()
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def _buildModel(self):
        print('build models....')
        # ==============set encoder head
        self._encoder_head = darknet.head2D(name='nolbo_encoder_head',
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=(2*3+3 + 2*(8+8)),
                                            filter_num_list=[1024, 1024, 1024],
                                            filter_size_list=[3, 3, 3],
                                            last_pooling='max', activation='elu')
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        self._priornet_cl = priornet.priornet(structure=self._prior_cl_str)
        self._priornet_inst = priornet.priornet(structure=self._prior_inst_str)
        print('done')

    def fit(self, inputs):
        class_list, inst_list, sin_gt, cos_gt, input_images, output_images_gt = inputs
        with tf.GradientTape() as tape:
            # get encoder output
            enc_output = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            inst_mean = enc_output[..., :8]
            inst_log_var = enc_output[..., 8:16]
            class_mean = enc_output[..., 16:16+8]
            class_log_var = enc_output[..., 16+8:16+16]
            sin_mean = tf.tanh(enc_output[..., 16+16: 16+16+3])
            cos_mean = tf.tanh(enc_output[..., 16+16+3:16+16+3+3])
            rad_log_var = enc_output[..., 16+16+3+3:]
            mean = tf.concat([inst_mean, class_mean], axis=-1)
            log_var = tf.concat([inst_log_var, class_log_var], axis=-1)
            latents = sampling(mu=mean, logVar=log_var)

            loss_sincos_kl, loss_sincos_mse, loss_sincos_1 = self._poseLoss(
                sin_gt=sin_gt, cos_gt=cos_gt, rad_var_gt=self._rad_var,
                sin=sin_mean, cos=cos_mean, rad_log_var=rad_log_var)

            inst_mean_prior, inst_log_var_prior = self._priornet_inst(tf.concat([class_list, inst_list], axis=-1), training=True)
            class_mean_prior, class_log_var_prior = self._priornet_cl(class_list, training=True)
            mean_prior = tf.concat([inst_mean_prior, class_mean_prior], axis=-1)
            log_var_prior = tf.concat([inst_log_var_prior, class_log_var_prior], axis=-1)
            output_images = self._decoder(latents, training=True)

            loss_shape = binary_loss(xPred=output_images, xTarget=output_images_gt, gamma=0.60)
            loss_latent_kl = kl_loss(mean=mean, logVar=log_var, mean_target=mean_prior, logVar_target=log_var_prior)
            loss_inst_prior_reg = regulizer_loss(z_mean=inst_mean_prior, z_logVar=inst_log_var_prior,
                                                  dist_in_z_space=5.0 * 8, class_input=class_list)
            loss_class_prior_reg = regulizer_loss(z_mean=class_mean_prior, z_logVar=class_log_var_prior,
                                                  dist_in_z_space=5.0 * 8)

            loss_sincos_kl = tf.nn.compute_average_loss(loss_sincos_kl, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_sincos_mse = tf.nn.compute_average_loss(loss_sincos_mse, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_sincos_1 = tf.nn.compute_average_loss(loss_sincos_1, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_shape = tf.nn.compute_average_loss(loss_shape, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_latent_kl = tf.nn.compute_average_loss(loss_latent_kl, global_batch_size=self._GLOBAL_BATCH_SIZE)
            loss_prior_reg = tf.nn.compute_average_loss(loss_inst_prior_reg+loss_class_prior_reg, global_batch_size=self._GLOBAL_BATCH_SIZE)

            total_loss = (
                loss_sincos_kl + 100.0 * loss_sincos_mse + 1000.0 * loss_sincos_1
                + loss_shape
                + loss_latent_kl
                + 0.01 * loss_prior_reg
            )
        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables \
                              + self._decoder.trainable_variables + self._priornet_inst.trainable_variables + self._priornet_cl.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images_gt, xPred=output_images)
        pr = tf.nn.compute_average_loss(TP / (TP + FP + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE)
        rc = tf.nn.compute_average_loss(TP / (TP + FN + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE)

        return loss_sincos_kl, loss_sincos_mse, loss_sincos_1,\
        loss_shape, loss_latent_kl, loss_prior_reg,\
        pr, rc

    def distributed_fit(self, inputs):
        sckl, scmse, sc1, s, lkl, reg, pr, rc = self._strategy.run(self.fit, args=(inputs,))
        sckl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, sckl, axis=None)
        scmse = self._strategy.reduce(tf.distribute.ReduceOp.SUM, scmse, axis=None)
        sc1 = self._strategy.reduce(tf.distribute.ReduceOp.SUM, sc1, axis=None)
        s = self._strategy.reduce(tf.distribute.ReduceOp.SUM, s, axis=None)
        lkl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lkl, axis=None)
        reg = self._strategy.reduce(tf.distribute.ReduceOp.SUM, reg, axis=None)

        pr = self._strategy.reduce(tf.distribute.ReduceOp.SUM, pr, axis=None)
        rc = self._strategy.reduce(tf.distribute.ReduceOp.SUM, rc, axis=None)
        return sckl, scmse, sc1, s, lkl, reg, pr, rc

    def saveEncoderBackbone(self, save_path):
        file_name = 'nolbo_encoder_backbone'
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))
    def saveEncoderHead(self, save_path):
        file_name = 'nolbo_encoder_head'
        self._encoder_head.save_weights(os.path.join(save_path, file_name))
    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)
    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))
    def savePriornet(self, save_path):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.save_weights(os.path.join(save_path, file_name_inst))
        self._priornet_cl.save_weights(os.path.join(save_path, file_name_class))
    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriornet(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = 'nolbo_encoder_backbone'
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))
    def loadEncoderHead(self, load_path, file_name=None):
        if file_name == None:
            file_name = 'nolbo_encoder_head'
        self._encoder_head.load_weights(os.path.join(load_path, file_name))
    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)
    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))
    def loadPriornet(self, load_path, file_name=None):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.load_weights(os.path.join(load_path, file_name_inst))
        self._priornet_cl.load_weights(os.path.join(load_path, file_name_class))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriornet(load_path=load_path)

    def _getEV(self, sin, cos, radLogVar):
        Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
        Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
        Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
            -tf.exp(radLogVar)) * sin * sin
        Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
            -tf.exp(radLogVar)) * cos * cos
        logVarsin = tf.math.log(Varsin + 1e-7)
        logVarcos = tf.math.log(Varcos + 1e-7)
        return Esin, Ecos, logVarsin, logVarcos

    def _poseLoss(self, sin_gt, cos_gt, rad_var_gt, sin, cos, rad_log_var):
        Esin_gt, Ecos_gt, log_var_sin_gt, log_var_cos_gt = self._getEV(
            sin=sin_gt, cos=cos_gt, radLogVar=tf.math.log(rad_var_gt+1e-7))
        Esin_pr, Ecos_pr, log_var_sin_pr, log_var_cos_pr = self._getEV(
            sin=sin, cos=cos, radLogVar=rad_log_var)

        loss_sin_kl = kl_loss(mean=Esin_pr, logVar=log_var_sin_pr, mean_target=Esin_gt, logVar_target=log_var_sin_gt)
        loss_cos_kl = kl_loss(mean=Ecos_pr, logVar=log_var_cos_pr, mean_target=Ecos_gt, logVar_target=log_var_cos_gt)

        sinz = sampling(mu=Esin_pr, logVar=log_var_sin_pr)
        cosz = sampling(mu=Ecos_pr, logVar=log_var_cos_pr)
        loss_sincos_mse = tf.square(sin_gt - sin)/tf.exp(log_var_sin_gt) \
                                + tf.square(cos_gt - cos)/tf.exp(log_var_cos_gt) \
                                + tf.square(rad_log_var - tf.math.log(rad_var_gt+1e-9)) \
                                + tf.square(sin_gt - sinz) + tf.square(cos_gt - cosz)
                                # + tf.square(Esin_gt - Esin_pr) + tf.square(Ecos_gt - Ecos_pr) \
                                # + tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile)+ tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        # self._loss_sincos_mse = tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile) \
        #                         + tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        #                         + tf.square(self._rad_log_var_tile - tf.math.log(self._rad_var+1e-9))
        loss_sincos_1 = tf.square(tf.square(sin)+tf.square(cos) - 1.0)

        return loss_sin_kl + loss_cos_kl, loss_sincos_mse, loss_sincos_1


class pretrain_integrated(object):
    def __init__(self,
                 backbone_style=None,
                 encoder_backbone=None,
                 decoder_structure=None,
                 prior_class_structure=None,
                 prior_inst_structure=None,
                 BATCH_SIZE_PER_REPLICA_nolbo=32,
                 BATCH_SIZE_PER_REPLICA_classifier=64,
                 strategy=None,
                 learning_rate = 1e-4
                 ):
        self._encoder_backbone = encoder_backbone
        self._backbone_style = backbone_style
        self._rad_var = (15.0/180.0 * 3.141593) ** 2
        self._dec_str = decoder_structure
        self._prior_cl_str = prior_class_structure
        self._prior_inst_str = prior_inst_structure

        self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        self._GLOBAL_BATCH_SIZE_nolbo = BATCH_SIZE_PER_REPLICA_nolbo * self._strategy.num_replicas_in_sync
        self._GLOBAL_BATCH_SIZE_classifier = BATCH_SIZE_PER_REPLICA_classifier * self._strategy.num_replicas_in_sync

        with self._strategy.scope():
            self._buildModel()
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
            # self._optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    def _buildModel(self):
        print('build models....')
        if self._encoder_backbone == None:
            self._encoder_backbone = self._backbone_style(name='backbone', activation='elu')

        # ================================= set model head
        self._encoder_head_imagenet = darknet.head2D(name='head_imagenet',
                                                  input_shape=self._encoder_backbone.output_shape[1:],
                                                  output_dim=1000,
                                                  filter_num_list=[],
                                                  filter_size_list=[],
                                                  last_pooling='max', activation='elu')
        self._encoder_head_place365 = darknet.head2D(name='head_imagenet',
                                                  input_shape=self._encoder_backbone.output_shape[1:],
                                                  output_dim=365,
                                                  filter_num_list=[],
                                                  filter_size_list=[],
                                                  last_pooling='max', activation='elu')

        # ==============set encoder head
        self._encoder_head_nolbo = darknet.head2D(name='head_nolbo',
                                            input_shape=self._encoder_backbone.output_shape[1:],
                                            output_dim=(2*3+3 + 2*(8+8)),
                                            filter_num_list=[1024, 1024, 1024],
                                            filter_size_list=[3, 3, 3],
                                            last_pooling='max', activation='elu')
        # ==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        self._priornet_cl = priornet.priornet(structure=self._prior_cl_str)
        self._priornet_inst = priornet.priornet(structure=self._prior_inst_str)
        print('done')

    # @tf.function
    def _lossObject(self, y_target, y_pred):
        y_pred = tf.nn.softmax(y_pred)
        loss = -tf.reduce_sum(y_target * tf.math.log(y_pred + 1e-9), axis=-1)
        return tf.nn.compute_average_loss(loss, global_batch_size=self._GLOBAL_BATCH_SIZE_classifier)

    # @tf.function
    def _evaluation(self, y_target, y_pred):
        gt = tf.argmax(y_target, axis=-1)
        pr = tf.argmax(y_pred, axis=-1)
        equality = tf.equal(pr, gt)
        acc_top1 = tf.cast(equality, tf.float32)
        acc_top5 = tf.cast(
            tf.math.in_top_k(
                predictions=y_pred,
                targets=gt, k=5
            ),
            tf.float32)
        return tf.nn.compute_average_loss(
            acc_top1, global_batch_size=self._GLOBAL_BATCH_SIZE_classifier
        ), tf.nn.compute_average_loss(
            acc_top5, global_batch_size=self._GLOBAL_BATCH_SIZE_classifier
        )

    def fit(self, inputs_imagenet, inputs_place365, inputs_nolbo):
        input_images_imagenet, class_list_imagenet = inputs_imagenet
        input_images_place365, class_list_place365 = inputs_place365
        class_list, inst_list, sin_gt, cos_gt, input_images, output_images_gt = inputs_nolbo
        with tf.GradientTape() as tape:
            class_list_imagenet_pred = self._encoder_head_imagenet(self._encoder_backbone(input_images_imagenet, training=True), training=True)
            pred_loss_imagenet = self._lossObject(y_target=class_list_imagenet, y_pred=class_list_imagenet_pred)
            class_list_place365_pred = self._encoder_head_place365(self._encoder_backbone(input_images_place365, training=True), training=True)
            pred_loss_place365 = self._lossObject(y_target=class_list_place365, y_pred=class_list_place365_pred)

            # get encoder output
            enc_output = self._encoder_head_nolbo(self._encoder_backbone(input_images, training=True), training=True)
            inst_mean = enc_output[..., :8]
            inst_log_var = enc_output[..., 8:16]
            class_mean = enc_output[..., 16:16+8]
            class_log_var = enc_output[..., 16+8:16+16]
            sin_mean = tf.tanh(enc_output[..., 16+16: 16+16+3])
            cos_mean = tf.tanh(enc_output[..., 16+16+3:16+16+3+3])
            rad_log_var = enc_output[..., 16+16+3+3:]
            mean = tf.concat([inst_mean, class_mean], axis=-1)
            log_var = tf.concat([inst_log_var, class_log_var], axis=-1)
            latents = sampling(mu=mean, logVar=log_var)

            loss_sincos_kl, loss_sincos_mse, loss_sincos_1 = self._poseLoss(
                sin_gt=sin_gt, cos_gt=cos_gt, rad_var_gt=self._rad_var,
                sin=sin_mean, cos=cos_mean, rad_log_var=rad_log_var)

            inst_mean_prior, inst_log_var_prior = self._priornet_inst(tf.concat([class_list, inst_list], axis=-1), training=True)
            class_mean_prior, class_log_var_prior = self._priornet_cl(class_list, training=True)
            mean_prior = tf.concat([inst_mean_prior, class_mean_prior], axis=-1)
            log_var_prior = tf.concat([inst_log_var_prior, class_log_var_prior], axis=-1)
            output_images = self._decoder(latents, training=True)

            loss_shape = binary_loss(xPred=output_images, xTarget=output_images_gt, gamma=0.60)
            loss_latent_kl = kl_loss(mean=mean, logVar=log_var, mean_target=mean_prior, logVar_target=log_var_prior)
            loss_inst_prior_reg = regulizer_loss(z_mean=inst_mean_prior, z_logVar=inst_log_var_prior,
                                                  dist_in_z_space=5.0 * 8, class_input=class_list)
            loss_class_prior_reg = regulizer_loss(z_mean=class_mean_prior, z_logVar=class_log_var_prior,
                                                  dist_in_z_space=5.0 * 8)

            loss_sincos_kl = tf.nn.compute_average_loss(loss_sincos_kl, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_sincos_mse = tf.nn.compute_average_loss(loss_sincos_mse, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_sincos_1 = tf.nn.compute_average_loss(loss_sincos_1, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_shape = tf.nn.compute_average_loss(loss_shape, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_latent_kl = tf.nn.compute_average_loss(loss_latent_kl, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
            loss_prior_reg = tf.nn.compute_average_loss(loss_inst_prior_reg+loss_class_prior_reg, global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)

            total_loss = (
                pred_loss_imagenet
                + pred_loss_place365
                + loss_sincos_kl + 100.0 * loss_sincos_mse + 1000.0 * loss_sincos_1
                + loss_shape
                + loss_latent_kl
                + 0.01 * loss_prior_reg
            )
        trainable_variables = self._encoder_backbone.trainable_variables\
                              + self._encoder_head_imagenet.trainable_variables + self._encoder_head_place365.trainable_variables + self._encoder_head_nolbo.trainable_variables \
                              + self._decoder.trainable_variables + self._priornet_inst.trainable_variables + self._priornet_cl.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        acc_top1_imagenet, acc_top5_imagenet = self._evaluation(y_target=class_list_imagenet, y_pred=class_list_imagenet_pred)
        acc_top1_place365, acc_top5_place365 = self._evaluation(y_target=class_list_place365, y_pred=class_list_place365_pred)

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images_gt, xPred=output_images)
        pr = tf.nn.compute_average_loss(TP / (TP + FP + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)
        rc = tf.nn.compute_average_loss(TP / (TP + FN + 1e-10), global_batch_size=self._GLOBAL_BATCH_SIZE_nolbo)

        return pred_loss_imagenet, pred_loss_place365, acc_top1_imagenet, acc_top1_place365, \
        acc_top5_imagenet, acc_top5_place365, loss_sincos_mse, loss_shape, pr, rc

    def distributed_fit(self, inputs_imagenet, inputs_place365, inputs_nolbo):
        limage, lplace, t1image, t1place, t5image, t5place, lscmse, lshape, pr, rc = self._strategy.run(self.fit, args=(inputs_imagenet, inputs_place365, inputs_nolbo,))
        limage = self._strategy.reduce(tf.distribute.ReduceOp.SUM, limage, axis=None)
        lplace = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lplace, axis=None)
        t1image = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t1image, axis=None)
        t1place = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t1place, axis=None)
        t5image = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t5image, axis=None)
        t5place = self._strategy.reduce(tf.distribute.ReduceOp.SUM, t5place, axis=None)
        lscmse = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lscmse, axis=None)
        lshape = self._strategy.reduce(tf.distribute.ReduceOp.SUM, lshape, axis=None)

        pr = self._strategy.reduce(tf.distribute.ReduceOp.SUM, pr, axis=None)
        rc = self._strategy.reduce(tf.distribute.ReduceOp.SUM, rc, axis=None)
        return limage, lplace, t1image, t1place, t5image, t5place, lscmse, lshape, pr, rc

    def saveEncoderBackbone(self, save_path):
        file_name = 'backbone'
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))
    def saveEncoderHead(self, save_path):
        self._encoder_head_imagenet.save_weights(os.path.join(save_path, 'head_imagenet'))
        self._encoder_head_place365.save_weights(os.path.join(save_path, 'head_place365'))
        self._encoder_head_nolbo.save_weights(os.path.join(save_path, 'head_nolbo'))
    def saveEncoder(self, save_path):
        self.saveEncoderBackbone(save_path=save_path)
        self.saveEncoderHead(save_path=save_path)
    def saveDecoder(self, save_path):
        file_name = self._dec_str['name']
        self._decoder.save_weights(os.path.join(save_path, file_name))
    def savePriornet(self, save_path):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.save_weights(os.path.join(save_path, file_name_inst))
        self._priornet_cl.save_weights(os.path.join(save_path, file_name_class))
    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriornet(save_path=save_path)

    def loadEncoderBackbone(self, load_path, file_name=None):
        if file_name == None:
            file_name = 'backbone'
        self._encoder_backbone.load_weights(os.path.join(load_path, file_name))
    def loadEncoderHead(self, load_path):
        self._encoder_head_imagenet.load_weights(os.path.join(load_path, 'head_imagenet'))
        self._encoder_head_place365.load_weights(os.path.join(load_path, 'head_place365'))
        self._encoder_head_nolbo.load_weights(os.path.join(load_path, 'head_nolbo'))
    def loadEncoder(self, load_path):
        self.loadEncoderBackbone(load_path=load_path)
        self.loadEncoderHead(load_path=load_path)
    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))
    def loadPriornet(self, load_path, file_name=None):
        file_name_inst = self._prior_inst_str['name']
        file_name_class = self._prior_cl_str['name']
        self._priornet_inst.load_weights(os.path.join(load_path, file_name_inst))
        self._priornet_cl.load_weights(os.path.join(load_path, file_name_class))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriornet(load_path=load_path)

    def _getEV(self, sin, cos, radLogVar):
        Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
        Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
        Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
            -tf.exp(radLogVar)) * sin * sin
        Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
            -tf.exp(radLogVar)) * cos * cos
        logVarsin = tf.math.log(Varsin + 1e-7)
        logVarcos = tf.math.log(Varcos + 1e-7)
        return Esin, Ecos, logVarsin, logVarcos

    def _poseLoss(self, sin_gt, cos_gt, rad_var_gt, sin, cos, rad_log_var):
        Esin_gt, Ecos_gt, log_var_sin_gt, log_var_cos_gt = self._getEV(
            sin=sin_gt, cos=cos_gt, radLogVar=tf.math.log(rad_var_gt+1e-9))
        Esin_pr, Ecos_pr, log_var_sin_pr, log_var_cos_pr = self._getEV(
            sin=sin, cos=cos, radLogVar=rad_log_var)

        loss_sin_kl = kl_loss(mean=Esin_pr, logVar=log_var_sin_pr, mean_target=Esin_gt, logVar_target=log_var_sin_gt)
        loss_cos_kl = kl_loss(mean=Ecos_pr, logVar=log_var_cos_pr, mean_target=Ecos_gt, logVar_target=log_var_cos_gt)

        sinz = sampling(mu=Esin_pr, logVar=log_var_sin_pr)
        cosz = sampling(mu=Ecos_pr, logVar=log_var_cos_pr)
        loss_sincos_mse = tf.square(sin_gt - sin)/tf.exp(log_var_sin_gt) \
                                + tf.square(cos_gt - cos)/tf.exp(log_var_cos_gt) \
                                + tf.square(rad_log_var - tf.math.log(rad_var_gt+1e-9)) \
                                + tf.square(sin_gt - sinz) + tf.square(cos_gt - cosz)
                                # + tf.square(Esin_gt - Esin_pr) + tf.square(Ecos_gt - Ecos_pr) \
                                # + tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile)+ tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        # self._loss_sincos_mse = tf.square(self._ori_sin_gt_tile - self._ori_sin_mean_tile) \
        #                         + tf.square(self._ori_cos_gt_tile - self._ori_cos_mean_tile) \
        #                         + tf.square(self._rad_log_var_tile - tf.math.log(self._rad_var+1e-9))
        loss_sincos_1 = tf.square(tf.square(sin)+tf.square(cos) - 1.0)

        return loss_sin_kl + loss_cos_kl, loss_sincos_mse, loss_sincos_1



















