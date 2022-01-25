import src.net_core.darknet as darknet
import src.net_core.autoencoder3D as ae3D
import src.net_core.priornet as priornet
import numpy as np

from src.module.function import *

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
                 backbone_style=None, encoder_backbone=None, strategy=None,
                 learning_rate=1e-4):
        self._enc_backbone_str = nolbo_structure['encoder_backbone']
        self._enc_head_str = nolbo_structure['encoder_head']
        self._dec_str = nolbo_structure['decoder']
        self._prior_str = nolbo_structure['prior']

        self._rad_var = (15.0/180.0 * 3.14159265358979323846) ** 2

        self._backbone_style = backbone_style
        self._encoder_backbone = encoder_backbone

        self._strategy = strategy
        # self._strategy = tf.distribute.MirroredStrategy()
        # self._BATCH_SIZE_PER_REPLICA = BATCH_SIZE_PER_REPLICA
        # self._GLOBAL_BATCH_SIZE = self._BATCH_SIZE_PER_REPLICA * self._strategy.num_replicas_in_sync

        with self._strategy.scope():
            self._buildModel()
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
        #==============set decoder3D
        self._decoder = ae3D.decoder3D(structure=self._dec_str)
        self._priornet = priornet.priornet(structure=self._prior_str)
        print('done')

    def _fit(self, encoder_inputs, decoder_inputs, encoder_batch_size_GLOBAL, decoder_batch_size_GLOBAL):
        offset_x, offset_y, input_images, objness_gt, bbox2D_gt, bbox3D_gt, ori_sin_gt, ori_cos_gt = encoder_inputs
        shape_masks, output_images_gt, inst_vectors_gt = decoder_inputs

        with tf.GradientTape() as tape:
            # ================  get encoder output ==================
            enc_output = self._encoder_head(self._encoder_backbone(input_images, training=True), training=True)
            objness, bbox2D, bbox3D, inst_mean, inst_log_var, ori_sin_mean, ori_cos_mean, rad_log_var \
                = self._encOutPartitioning(enc_output)

            # =============== get tiles of encoder output and obj_mask
            objness_tile, bbox2D_tile, bbox3D_tile, inst_mean_tile, inst_log_var_tile, ori_sin_mean_tile, ori_cos_mean_tile, rad_log_var_tile, \
            objness_gt_tile, bbox2D_gt_tile, bbox3D_gt_tile, ori_sin_gt_tile, ori_cos_gt_tile\
                = self._createTiles(
                objness, bbox2D, bbox3D, inst_mean, inst_log_var, ori_sin_mean, ori_cos_mean, rad_log_var,
                objness_gt, bbox2D_gt, bbox3D_gt, ori_cos_gt, ori_sin_gt)
            IOU = self._getbbox2DIOU(offset_x, offset_y, bbox2D_gt_tile, bbox2D_tile)
            obj_mask, objness_gt_with_IOU = self._getObjMaskAndObjGT(IOU, objness, objness_gt_tile)

            self._getTilesAndObjMask(enc_output, offset_x, offset_y, objness_gt, bbox2D_gt, bbox3D_gt, ori_cos_gt, ori_sin_gt)

            # ================ get encoder loss ================================
            loss_objness, loss_no_objness = self._objnessLoss(objness_gt_with_IOU, objness)
            loss_bbox2D_hw, loss_bbox2D_xy = self._bbox2DLoss(obj_mask, bbox2D_tile, bbox2D_gt_tile)
            loss_bbox3D = self._bbox3DLoss(obj_mask, bbox3D_tile, bbox3D_gt_tile)
            loss_sincos_mse, loss_sincos_1, loss_sincos_kl = self._poseLoss(
                obj_mask, ori_sin_mean_tile, ori_cos_mean_tile, rad_log_var_tile,
                ori_sin_gt_tile, ori_cos_gt_tile)

            # ================= select predicted objects =======================
            obj_mask = tf.cast(tf.reshape(obj_mask, tf.shape(obj_mask)[:-1]), tf.int32)
            inst_mean_sel = tf.dynamic_partition(inst_mean_tile, obj_mask, 2)[1]
            inst_log_var_sel = tf.dynamic_partition(inst_log_var_tile, obj_mask, 2)[1]

            # ================= select gt objects =========================
            shape_masks = tf.cast(shape_masks, tf.int32)
            output_images_gt = tf.dynamic_partition(output_images_gt, shape_masks, 2)[1]
            inst_vectors_gt = tf.dynamic_partition(inst_vectors_gt, shape_masks, 2)[1]

            # ================ get (priornet, decoder) output and loss
            z = sampling(mu=inst_mean_sel, logVar=inst_log_var_sel)
            inst_mean_prior, inst_log_var_prior = self._priornet(inst_vectors_gt, training=True)
            outputs = self._decoder(z, training=True)
            loss_latents_kl = kl_loss(mean=inst_mean_sel, logVar=inst_log_var_sel, mean_target=inst_mean_prior, logVar_target=inst_log_var_prior)
            loss_shape = binary_loss(xPred=outputs, xTarget=output_images_gt, gamma=0.60, b_range=False)
            loss_prior_reg = regulizer_loss(z_mean=inst_mean_prior, z_logVar=inst_log_var_prior, dist_in_z_space=3.0 * self._enc_backbone_str['z_inst_dim'])

            # ================= get network parameter regulization loss
            reg_loss = tf.reduce_sum(self._encoder_head.losses + self._encoder_backbone.losses
                                     + self._priornet.losses + self._decoder.losses)

            # ================== take average of losses of replicas
            loss_objness = tf.nn.compute_average_loss(loss_objness, global_batch_size=encoder_batch_size_GLOBAL)
            loss_no_objness = tf.nn.compute_average_loss(loss_no_objness, global_batch_size=encoder_batch_size_GLOBAL)
            loss_bbox2D_hw = tf.nn.compute_average_loss(loss_bbox2D_hw, global_batch_size=encoder_batch_size_GLOBAL)
            loss_bbox2D_xy = tf.nn.compute_average_loss(loss_bbox2D_xy, global_batch_size=encoder_batch_size_GLOBAL)
            loss_bbox3D = tf.nn.compute_average_loss(loss_bbox3D, global_batch_size=encoder_batch_size_GLOBAL)
            loss_sincos_mse = tf.nn.compute_average_loss(loss_sincos_mse, global_batch_size=encoder_batch_size_GLOBAL)
            loss_sincos_1 = tf.nn.compute_average_loss(loss_sincos_1, global_batch_size=encoder_batch_size_GLOBAL)
            loss_sincos_kl = tf.nn.compute_average_loss(loss_sincos_kl, global_batch_size=encoder_batch_size_GLOBAL)
            loss_shape = tf.nn.compute_average_loss(loss_shape, global_batch_size=decoder_batch_size_GLOBAL)
            loss_latents_kl = tf.nn.compute_average_loss(loss_latents_kl, global_batch_size=decoder_batch_size_GLOBAL)
            loss_prior_reg = tf.nn.compute_average_loss(loss_prior_reg, global_batch_size=decoder_batch_size_GLOBAL)

            # =================== total loss
            total_loss = (
                    20.0 * loss_objness + 0.5 * loss_no_objness
                    + 5.0 * (loss_bbox2D_hw + loss_bbox2D_xy)
                    + loss_bbox3D
                    + loss_shape + loss_latents_kl
                    + 0.01 * loss_prior_reg
                    + 100.0 * loss_sincos_mse + 100.0 * loss_sincos_1
                    + loss_sincos_kl
                    + reg_loss
            )

        trainable_variables = self._encoder_backbone.trainable_variables + self._encoder_head.trainable_variables\
                              + self._decoder.trainable_variables + self._priornet.trainable_variables
        grads = tape.gradient(total_loss, trainable_variables)
        self._optimizer.apply_gradients(zip(grads, trainable_variables))

        TP, FP, FN = voxelPrecisionRecall(xTarget=output_images_gt, xPred=outputs)
        TP = tf.nn.compute_average_loss(TP, global_batch_size=decoder_batch_size_GLOBAL)
        FP = tf.nn.compute_average_loss(FP, global_batch_size=decoder_batch_size_GLOBAL)
        FN = tf.nn.compute_average_loss(FN, global_batch_size=decoder_batch_size_GLOBAL)

        return loss_objness, loss_no_objness, loss_bbox2D_hw, loss_bbox2D_xy, loss_bbox3D, \
               loss_sincos_mse, loss_sincos_1, loss_sincos_kl, \
               loss_shape, loss_latents_kl, loss_prior_reg, \
               TP, FP, FN

    @tf.function
    def distributed_fit(self, encoder_inputs, decoder_inputs, encoder_batch_size_GLOBAL, decoder_batch_size_GLOBAL):
        l_obj, l_noobj, l_b2Dhw, l_b2Dxy, l_b3D, \
        l_scmse, l_lsc1, l_sckl, \
        l_sh, l_lkl, l_pr, TP, FP, FN \
            = self._strategy.run(self._fit,
                                 args=(next(iter(encoder_inputs)), next(iter(decoder_inputs)),
                                       encoder_batch_size_GLOBAL, decoder_batch_size_GLOBAL,))
        l_obj = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_obj, axis=None)
        l_noobj = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_noobj, axis=None)
        l_b2Dhw = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_b2Dhw, axis=None)
        l_b2Dxy = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_b2Dxy, axis=None)
        l_b3D = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_b3D, axis=None)
        l_scmse = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_scmse, axis=None)
        l_lsc1 = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_lsc1, axis=None)
        l_sckl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_sckl, axis=None)
        l_lkl = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_lkl, axis=None)
        l_sh = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_sh, axis=None)
        l_pr = self._strategy.reduce(tf.distribute.ReduceOp.SUM, l_pr, axis=None)
        TP = self._strategy.reduce(tf.distribute.ReduceOp.SUM, TP, axis=None)
        FP = self._strategy.reduce(tf.distribute.ReduceOp.SUM, FP, axis=None)
        FN = self._strategy.reduce(tf.distribute.ReduceOp.SUM, FN, axis=None)
        pr = TP / (TP + FP + 1e-10)
        rc = TP / (TP + FN + 1e-10)
        return l_obj, l_noobj, l_b2Dhw, l_b2Dxy, l_b3D, l_scmse, l_lsc1, l_sckl, l_sh, l_lkl, l_pr, pr, rc

    def saveEncoderBackbone(self, save_path):
        file_name = self._enc_backbone_str['name']
        self._encoder_backbone.save_weights(os.path.join(save_path, file_name))
    def saveEncoderHead(self, save_path):
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
        self._priornet.save_weights(os.path.join(save_path, file_name))
    def saveModel(self, save_path):
        self.saveEncoder(save_path=save_path)
        self.saveDecoder(save_path=save_path)
        self.savePriornet(save_path=save_path)

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
    def loadDecoder(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._dec_str['name']
        self._decoder.load_weights(os.path.join(load_path, file_name))
    def loadPriornet(self, load_path, file_name=None):
        if file_name == None:
            file_name = self._prior_str['name']
        self._priornet.load_weights(os.path.join(load_path, file_name))
    def loadModel(self, load_path):
        self.loadEncoder(load_path=load_path)
        self.loadDecoder(load_path=load_path)
        self.loadPriornet(load_path=load_path)

    def _getTilesAndObjMask(self, enc_output, offset_x, offset_y, objness_gt, bbox2D_gt, bbox3D_gt, ori_cos_gt, ori_sin_gt):
        objness, bbox2D, bbox3D, inst_mean, inst_log_var, ori_sin_mean, ori_cos_mean, rad_log_var = self._encOutPartitioning(enc_output)
        objness_tile, bbox2D_tile, bbox3D_tile, inst_mean_tile, inst_log_var_tile, ori_sin_mean_tile, ori_cos_mean_tile, rad_log_var_tile, \
        objness_gt_tile, bbox2D_gt_tile, bbox3D_gt_tile, ori_sin_gt_tile, ori_cos_gt_tile = self._createTiles(
            objness, bbox2D, bbox3D, inst_mean, inst_log_var, ori_sin_mean, ori_cos_mean, rad_log_var,
                     objness_gt, bbox2D_gt, bbox3D_gt, ori_cos_gt, ori_sin_gt)
        IOU = self._getbbox2DIOU(offset_x, offset_y, bbox2D_gt_tile, bbox2D_tile)
        obj_mask, objness_gt_with_IOU = self._getObjMaskAndObjGT(IOU, objness, objness_gt_tile)
        return obj_mask, objness_gt_with_IOU,

    def _encOutPartitioning(self, enc_output):
        pr_num = self._enc_backbone_str['predictor_num']
        objness, bbox2D, bbox3D = [], [], []
        inst_mean, inst_log_var = [], []
        ori_sin_mean, ori_cos_mean, rad_log_var = [], [], []
        part_start = 0
        part_end = part_start
        for predIndex in range(pr_num):
            # objectness
            part_end += 1
            objness.append(enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox2D_dim']
            bbox2D.append(enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['bbox3D_dim']
            bbox3D.append(enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['z_inst_dim']
            inst_mean.append(enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['z_inst_dim']
            inst_log_var.append(enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            ori_sin_mean.append(enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            ori_cos_mean.append(enc_output[..., part_start:part_end])
            part_start = part_end
            part_end += self._enc_backbone_str['orientation_dim']
            rad_log_var.append(enc_output[..., part_start:part_end])
            part_start = part_end
        objness = tf.sigmoid(tf.transpose(tf.stack(objness), [1, 2, 3, 0, 4]))
        bbox2D = tf.sigmoid(tf.transpose(tf.stack(bbox2D), [1, 2, 3, 0, 4]))
        bbox3D = tf.transpose(tf.stack(bbox3D), [1,2,3,0,4])
        inst_mean = tf.transpose(tf.stack(inst_mean), [1, 2, 3, 0, 4])
        inst_log_var = tf.transpose(tf.stack(inst_log_var), [1, 2, 3, 0, 4])
        ori_sin_mean = -1.+2.*tf.sigmoid(tf.transpose(tf.stack(ori_sin_mean), [1, 2, 3, 0, 4]))
        ori_cos_mean = -1.+2.*tf.sigmoid(tf.transpose(tf.stack(ori_cos_mean), [1, 2, 3, 0, 4]))
        rad_log_var = tf.transpose(tf.stack(rad_log_var), [1, 2, 3, 0, 4])
        return objness, bbox2D, bbox3D, inst_mean, inst_log_var, ori_sin_mean, ori_cos_mean, rad_log_var

    def _createTiles(self, objness, bbox2D, bbox3D, inst_mean, inst_log_var, ori_sin_mean, ori_cos_mean, rad_log_var,
                     objness_gt, bbox2D_gt, bbox3D_gt, ori_cos_gt, ori_sin_gt):
        pr_num = self._enc_backbone_str['predictor_num']
        # tiles for prediction
        objness_tile = tf.tile(objness, [1, 1, 1, pr_num, 1])
        bbox2D_tile = tf.tile(bbox2D, [1, 1, 1, pr_num, 1])  # [batchSize,row,col,2*predNum,hwxy]
        bbox3D_tile = tf.tile(bbox3D, [1, 1, 1, pr_num, 1])
        inst_mean_tile = tf.tile(inst_mean, [1, 1, 1, pr_num, 1])
        inst_log_var_tile = tf.tile(inst_log_var, [1, 1, 1, pr_num, 1])
        ori_sin_mean_tile = tf.tile(ori_sin_mean, [1, 1, 1, pr_num, 1])
        ori_cos_mean_tile = tf.tile(ori_cos_mean, [1, 1, 1, pr_num, 1])
        rad_log_var_tile = tf.tile(rad_log_var, [1, 1, 1, pr_num, 1])
        # tiles for gt
        bbox2D_gt_tile, bbox3D_gt_tile, objness_gt_tile = [], [], []
        ori_cos_gt_tile, ori_sin_gt_tile = [], []
        for obj_index in range(pr_num):
            for pred_index in range(pr_num):
                objness_gt_tile += [objness_gt[:, :, :, obj_index:obj_index+1, :]]
                bbox2D_gt_tile += [bbox2D_gt[:, :, :, obj_index:obj_index+1, :]]
                bbox3D_gt_tile += [bbox3D_gt[:, :, :, obj_index:obj_index+1, :]]
                ori_cos_gt_tile += [ori_cos_gt[:, :, :, obj_index:obj_index+1, :]]
                ori_sin_gt_tile += [ori_sin_gt[:, :, :, obj_index:obj_index+1, :]]
        objness_gt_tile = tf.concat(objness_gt_tile, axis=-2)
        bbox2D_gt_tile = tf.concat(bbox2D_gt_tile, axis=-2)
        bbox3D_gt_tile = tf.concat(bbox3D_gt_tile, axis=-2)
        ori_sin_gt_tile = tf.concat(ori_sin_gt_tile, axis=-2)
        ori_cos_gt_tile = tf.concat(ori_cos_gt_tile, axis=-2)
        return objness_tile, bbox2D_tile, bbox3D_tile, inst_mean_tile, inst_log_var_tile, ori_sin_mean_tile, ori_cos_mean_tile, rad_log_var_tile,\
    objness_gt_tile, bbox2D_gt_tile, bbox3D_gt_tile, ori_sin_gt_tile, ori_cos_gt_tile

    def _getbbox2DIOU(self, offset_x, offset_y, bbox2D_gt_tile, bbox2D_tile):
        pr_num = self._enc_backbone_str['predictor_num']
        offset_x_tile = tf.tile(offset_x, [1, 1, 1, pr_num])
        offset_y_tile = tf.tile(offset_y, [1, 1, 1, pr_num])
        len_grid_x, len_grid_y = tf.cast(tf.shape(offset_x)[2], tf.float32), tf.cast(tf.shape(offset_x)[1], tf.float32)
        # tf.shape(offset) = [batch, Y, X] ([0,1,2])
        box_gt = tf.stack([
            (bbox2D_gt_tile[..., 2] + offset_x_tile) - bbox2D_gt_tile[..., 1] / 2.0 * len_grid_x,  # x-W/2
            (bbox2D_gt_tile[..., 3] + offset_y_tile) - bbox2D_gt_tile[..., 0] / 2.0 * len_grid_y,  # y-H/2
            (bbox2D_gt_tile[..., 2] + offset_x_tile) - bbox2D_gt_tile[..., 1] / 2.0 * len_grid_x,  # x+W/2
            (bbox2D_gt_tile[..., 3] + offset_y_tile) - bbox2D_gt_tile[..., 0] / 2.0 * len_grid_y  # y+H/2
        ], axis=-1)
        box_pr = tf.stack([
            (bbox2D_tile[..., 2] + offset_x_tile) - bbox2D_tile[..., 1] / 2.0 * len_grid_x,
            (bbox2D_tile[..., 3] + offset_y_tile) - bbox2D_tile[..., 0] / 2.0 * len_grid_y,
            (bbox2D_tile[..., 2] + offset_x_tile) - bbox2D_tile[..., 1] / 2.0 * len_grid_x,
            (bbox2D_tile[..., 3] + offset_y_tile) - bbox2D_tile[..., 0] / 2.0 * len_grid_y
        ], axis=-1)
        left_up = tf.maximum(box_gt[..., :2], box_pr[..., :2]) # x-W/2, y-H/2
        right_down = tf.minimum(box_gt[..., :2], box_pr[..., :2]) # x+W/2, y+H/2
        intersection = tf.maximum(0., right_down - left_up)
        intersection_area = intersection[..., 0] * intersection[..., 1]
        box_gt_area = (box_gt[..., 2] - box_gt[..., 0]) * (box_gt[..., 3] - box_gt[..., 1])
        box_pr_area = (box_pr[..., 2] - box_pr[..., 0]) * (box_pr[..., 3] - box_pr[..., 1])
        union_area = tf.maximum(box_gt_area + box_pr_area - intersection_area, 1e-9)
        IOU = tf.clip_by_value(intersection_area/union_area, 0., 1.)
        return IOU

    def _getObjMaskAndObjGT(self, IOU, objness, objness_gt_tile):
        pr_num = self._enc_backbone_str['predictor_num']
        obj_mask_arr = []
        used_predictor_mask = tf.zeros_like(IOU[...,0:pr_num])
        for obj_index in range(pr_num):
            IOU_per_obj = IOU[..., obj_index*pr_num:(obj_index+1)*pr_num]
            IOU_per_obj = IOU_per_obj * (1.0 - used_predictor_mask) * objness[..., 0]
            objness_gt_per_obj = objness_gt_tile[..., obj_index*pr_num:(obj_index+1)*pr_num, 0]
            _, indices = tf.nn.top_k(IOU_per_obj, k=pr_num)
            indices = tf.map_fn(tf.math.invert_permutation, tf.reshape(indices, (-1,pr_num)))
            indices = tf.reshape(indices, tf.shape(used_predictor_mask))
            mask_per_obj = tf.cast(tf.where(indices<=0, tf.ones_like(indices), tf.zeros_like(indices)), tf.float32)
            mask_per_obj = mask_per_obj * objness_gt_per_obj
            used_predictor_mask = used_predictor_mask + mask_per_obj
            used_predictor_mask = tf.where(used_predictor_mask>0, tf.ones_like(used_predictor_mask), tf.zeros_like(used_predictor_mask))
            obj_mask_arr += [mask_per_obj]
        objness_gt_with_IOU = tf.stack(obj_mask_arr, axis=-1)
        objness_gt_with_IOU = tf.reduce_sum(objness_gt_with_IOU, axis=-1)
        objness_gt_with_IOU = tf.where(objness_gt_with_IOU>0, tf.ones_like(objness_gt_with_IOU), tf.zeros_like(objness_gt_with_IOU))
        objness_gt_with_IOU = tf.cast(objness_gt_with_IOU, tf.float32)
        obj_mask = tf.concat(obj_mask_arr, axis=-1)

        # add additional dimension : (:,:) -> (:,:,1)
        objness_gt_with_IOU = tf.stack([objness_gt_with_IOU], axis=-1)
        obj_mask = tf.stack([obj_mask], axis=-1)
        return obj_mask, objness_gt_with_IOU

    def _objnessLoss(self, objness_gt_with_IOU, objness):
        d_objness = - objness_gt_with_IOU * tf.math.log(objness + 1e-10)
        d_no_objness = -(1.0-objness_gt_with_IOU) * tf.math.log(1.0 - objness + 1e-10)

        loss_objness = tf.reduce_sum(d_objness, axis=[1, 2, 3, 4])
        loss_no_objness = tf.reduce_sum(d_no_objness, axis=[1, 2, 3, 4])
        return loss_objness, loss_no_objness

    def _bbox2DLoss(self, obj_mask, bbox2D_tile, bbox2D_gt_tile):
        obj_mask = tf.reshape(obj_mask, tf.shape(obj_mask)[:-1])
        # tile shape = (batch, gridy, gridx, 2*predictornum, hwxy)
        d_x = obj_mask * (bbox2D_tile[..., 2] - bbox2D_gt_tile[..., 2])
        d_y = obj_mask * (bbox2D_tile[..., 3] - bbox2D_gt_tile[..., 3])
        d_h = obj_mask * (tf.sqrt(bbox2D_tile[..., 0]) - tf.sqrt(bbox2D_gt_tile[..., 0]))
        d_w = obj_mask * (tf.sqrt(bbox2D_tile[..., 1]) - tf.sqrt(bbox2D_gt_tile[..., 1]))
        loss_bbox2D_xy = tf.reduce_sum(tf.square(d_x) + tf.square(d_y), axis=[1, 2, 3])
        loss_bbox2D_hw = tf.reduce_sum(tf.square(d_h) + tf.square(d_w), axis=[1, 2, 3])
        return loss_bbox2D_hw, loss_bbox2D_xy

    def _bbox3DLoss(self, obj_mask, bbox3D_tile, bbox3D_gt_tile):
        # obj_mask = tf.reshape(self._obj_mask, tf.shape(self._obj_mask)[:-1])
        d = obj_mask * (bbox3D_tile - bbox3D_gt_tile)
        # d shape = (batch, gridy, gridx, 2*predictornum, whl)
        loss_bbox3D = tf.reduce_sum(tf.square(d), axis=[1, 2, 3, 4])
        return loss_bbox3D



    def _poseLoss(self, obj_mask, ori_sin_mean_tile, ori_cos_mean_tile, rad_log_var_tile, ori_sin_gt_tile, ori_cos_gt_tile):
        def getEV(sin, cos, radLogVar):
            Esin = tf.exp(-tf.exp(radLogVar) / 2.0) * sin
            Ecos = tf.exp(-tf.exp(radLogVar) / 2.0) * cos
            Varsin = 0.5 - 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (1.0 - 2.0 * sin * sin) - tf.exp(
                -tf.exp(radLogVar)) * sin * sin
            Varcos = 0.5 + 0.5 * tf.exp(-2.0 * tf.exp(radLogVar)) * (2.0 * cos * cos - 1.0) - tf.exp(
                -tf.exp(radLogVar)) * cos * cos
            logVarsin = tf.math.log(Varsin + 1e-7)
            logVarcos = tf.math.log(Varcos + 1e-7)
            return Esin, Ecos, logVarsin, logVarcos
        Esin_gt, Ecos_gt, log_var_sin_gt, log_var_cos_gt = getEV(sin=ori_sin_gt_tile, cos=ori_cos_gt_tile, radLogVar=tf.math.log(self._rad_var))
        Esin_pr, Ecos_pr, log_var_sin_pr, log_var_cos_pr = getEV(sin=ori_sin_mean_tile, cos=ori_cos_mean_tile, radLogVar=rad_log_var_tile)

        loss_sin_kl = kl_loss(mean=Esin_pr, logVar=log_var_sin_pr, mean_target=Esin_gt, logVar_target=log_var_sin_gt)
        loss_cos_kl = kl_loss(mean=Ecos_pr, logVar=log_var_cos_pr, mean_target=Ecos_gt, logVar_target=log_var_cos_gt)
        loss_sincos_mse = tf.square((ori_sin_gt_tile-ori_sin_mean_tile)/tf.sqrt(tf.exp(log_var_sin_gt))) + \
            tf.square((ori_cos_gt_tile-ori_cos_mean_tile)/tf.sqrt(tf.exp(log_var_cos_gt)))
        loss_sincos_1 = tf.square(tf.square(ori_sin_mean_tile)+tf.square(ori_cos_mean_tile) - 1.0)
        loss_sincos_mse = tf.reduce_sum(obj_mask * loss_sincos_mse, axis=[1, 2, 3, 4])
        loss_sincos_1 = tf.reduce_sum(obj_mask * loss_sincos_1, axis=[1, 2, 3, 4])

        obj_mask = tf.reshape(obj_mask, tf.shape(obj_mask)[:-1])
        loss_sin_kl = tf.reduce_sum(obj_mask * loss_sin_kl, axis=[1, 2, 3])
        loss_cos_kl = tf.reduce_sum(obj_mask * loss_cos_kl, axis=[1, 2, 3])
        loss_sincos_kl = loss_sin_kl + loss_cos_kl

        return loss_sincos_mse, loss_sincos_1, loss_sincos_kl

