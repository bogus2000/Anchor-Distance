import tensorflow as tf

tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')
import numpy as np

def box_intersection_th(corners1:tf.Tensor, corners2:tf.Tensor):
    '''
    :param corners1: (tf.Tensor) : (batch, gr, gc, pr, 4, 2)
    :param corners2: (tf.Tensor) : (batch, gr, gc, pr, 4, 2)
    :return: (tf.Tensor) : (batch, gr, gc, pr, 4, 4, 2)
    '''
    batch, gr, gc, pr, _, _ = corners1.get_shape().as_list()
    # print(batch, gr,gc,pr)
    reorder = tf.reshape(tf.convert_to_tensor(np.array([1,2,3,0] * batch * gr * gc * pr)), [batch,gr,gc,pr,4])

    corners1_reorder = tf.gather(corners1, reorder, batch_dims=4)
    corners2_reorder = tf.gather(corners2, reorder, batch_dims=4)
    # print(corners1_reorder.shape)
    # print(corners2_reorder.shape)
    line1 = tf.concat([corners1, corners1_reorder], axis=-1) # (b, gr,gc,pr, 4, 4)
    line2 = tf.concat([corners2, corners2_reorder], axis=-1)
    line1_ext = tf.tile(tf.expand_dims(line1, axis=5), [1,1,1,1,1,4,1]) # (b,gr,gc,pr,4,4,4)
    line2_ext = tf.tile(tf.expand_dims(line2, axis=4), [1,1,1,1,4,1,1]) # (b,gr,gc,pr,4,4,4)
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = den_t / (num + 1e-8)
    # t[num == .0] = -1.
    t = tf.where(num == 0., -1., t)
    # mask_t = (t > 0.) * (t < 1.)  # intersection on line segment 1
    mask_t = tf.where(t>0., 1., 0.) * tf.where(t<1., 1., 0.)
    den_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
    u = -den_u / num
    # u[num == .0] = -1.
    u = tf.where(num==0., -1, u)
    # mask_u = (u > 0.) * (u < 1.)  # intersection on line segment 2
    mask_u = tf.where(u>0., 1., 0.) * tf.where(u<1., 1., 0.)
    mask = mask_t * mask_u
    t = den_t / (num + 1e-8)  # overwrite with EPSILON. otherwise numerically unstable
    intersection = tf.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], axis=-1)
    intersection = intersection * tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
    return intersection, mask

def box1_in_box2(corners1:tf.Tensor, corners2:tf.Tensor):
    """check if corners of box1 lie in box2
        Convention: if a corner is exactly on the edge of the other box, it's also a valid point
        Args:
            corners1 (torch.Tensor): (batch,gr,gc,pr, 4, 2)
            corners2 (torch.Tensor): (batch,gr,gc,pr, 4, 2)
        Returns:
            c1_in_2: (batch,gr,gc,pr, 4) Bool
        """
    a = corners2[:,:,:,:,0:1, :] # (batch,gr,gc,pr,1,2)
    b = corners2[:,:,:,:,1:2, :] # (batch,gr,gc,pr,1,2)
    d = corners2[:,:,:,:,3:4, :] # (batch,gr,gc,pr,1,2)
    ab = b - a                 # (batch,gr,gc,pr,1,2)
    am = corners1 - a          # (batch,gr,gc,pr,4,2)
    ad = d - a                 # (batch,gr,gc,pr,1,2)
    p_ab = tf.reduce_sum(ab * am, axis=-1) # (batch,gr,gc,pr,4)
    norm_ab = tf.reduce_sum(ab*ab, axis=-1)# (batch,gr,gc,pr,1)
    p_ad = tf.reduce_sum(ad*am, axis=-1)   # (batch,gr,gc,pr,4)
    norm_ad = tf.reduce_sum(ad*ad, axis=-1)# (batch,gr,gc,pr,1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    # cond1 = (p_ab / norm_ab > - 1e-6) * (p_ab / norm_ab < 1 + 1e-6)  # (b,gr,gc,pr, 4)
    cond1 = tf.where(p_ab/(norm_ab+1e-8) > -1e-6, 1., 0.) * tf.where(p_ab/(norm_ab+1e-8) < 1+1e-6, 1., 0.)
    # cond2 = (p_ad / norm_ad > - 1e-6) * (p_ad / norm_ad < 1 + 1e-6)  # (b,gr,gc,pr, 4)
    cond2 = tf.where(p_ad/(norm_ad+1e-8) > -1e-6, 1., 0.) * tf.where(p_ad/(norm_ad+1e-8) < 1+1e-6, 1., 0.)
    return tf.constant(cond1 * cond2)

def box_in_box_th(corners1:tf.Tensor, corners2:tf.Tensor):
    """
    check if corners of two boxes lie in each other
    Args:
        corners1 (torch.Tensor): (b,gr,gc,pr, 4, 2)
        corners2 (torch.Tensor): (b,gr,gc,pr, 2)
    Returns:
        c1_in_2: (b,gr,gc,pr, 4) Bool. i-th corner of box1 in box2
        c2_in_1: (b,gr,gc,pr, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1

def build_vertices(corners1:tf.Tensor, corners2:tf.Tensor,
                   c1_in_2:tf.Tensor, c2_in_1:tf.Tensor,
                   inters:tf.Tensor, mask_inter:tf.Tensor):
    """find vertices of intersection area
        Args:
            corners1 (torch.Tensor): (batch,gr,gc,pr, 4, 2)
            corners2 (torch.Tensor): (batch,gr,gc,pr, 4, 2)
            c1_in_2 (torch.Tensor): Bool, (batch,gr,gc,pr, 4)
            c2_in_1 (torch.Tensor): Bool, (batch,gr,gc,pr, 4)
            inters (torch.Tensor): (batch,gr,gc,pr, 4, 4, 2)
            mask_inter (torch.Tensor): (batch,gr,gc,pr, 4, 4)

        Returns:
            vertices (torch.Tensor): (batch,gr,gc,pr, 24, 2) vertices of intersection area. only some elements are valid
            mask (torch.Tensor): (batch,gr,gc,pr, 24) indicates valid elements in vertices
        """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0).
    # can be used as trick
    batch, gr, gc,pr, _, _ = tf.shape(corners1)
    vertices = tf.concat([corners1, corners2, tf.reshape(inters, [batch,gr,gc,pr,-1,2])], axis=-2) # (batch,gr,gc,4+4+4*4, 2)
    mask = tf.concat([c1_in_2, c2_in_1, tf.reshape(mask_inter, [batch,gr,gc,pr,-1])], axis=-1) # (batch,gr,gc,4+4+4*4)
    return vertices, tf.constant(mask)


def sort_indices(vertices: tf.Tensor, mask: tf.Tensor):
    """[summary]
    Args:
        vertices (torch.Tensor): float (b,gr,gc,pr, 24, 2)
        mask (torch.Tensor): bool (b,gr,gc,pr, 24)
    Returns:
        sorted_index: bool (b,gr,gc,pr, 9)

    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X)
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    mask = tf.cast(mask, tf.float32)
    num_valid = tf.reduce_sum(mask, axis=-1) # (b,gr,gc,pr)
    mean = tf.reduce_sum(vertices * tf.expand_dims(mask, axis=-1), axis=-2) / (tf.expand_dims(num_valid, axis=-1) + 1e-8)# (b,gr,gc,pr,2)
    mean = tf.tile(tf.expand_dims(mean, axis=4), [1,1,1,1,24,1]) # (b,gr,gc,pr,24,2)
    vertices_normalized = vertices - mean # (b,gr,gc,pr,24,2)

    # sort counter-clockwise
    theta = tf.atan2(vertices_normalized[...,1], vertices_normalized[...,0]+1e-9) #(b,gr,gc,pr,24)
    theta = tf.where(theta<0., 3.14159 * 2 + theta, theta)
    # print(tf.reduce_max(theta))
    # print(tf.reduce_min(theta))
    # for ascending sort, high value for zero mask
    theta_ascending = tf.where(mask==0., 99.99 * tf.ones_like(theta), theta)
    idx_ascending = tf.argsort(theta_ascending, axis=-1, direction='ASCENDING') # (1,2,3,...,9999) (b,gr,gc,pr,24)
    # for descending, low value for zero mask
    theta_descending = tf.where(mask==0., -99.99 * tf.ones_like(theta), theta)
    idx_descending = tf.argsort(theta_descending, axis=-1, direction='DESCENDING') # (2 * 3.14, ..., -9999.9) (b,gr,gc,pr,24)

    vertices_sorted_ascending = tf.gather(params=vertices_normalized, indices=idx_ascending, batch_dims=4)   # (b,gr,gc,pr,24, 2)
    vertices_sorted_descending = tf.gather(params=vertices_normalized, indices=idx_descending, batch_dims=4) # (b,gr,gc,pr,24, 2)
    mask_sorted_ascending = tf.gather(params=mask, indices=idx_ascending, batch_dims=4) # (b,gr,gc,pr,24)
    mask_sorted_descending = tf.gather(params=mask, indices=idx_descending, batch_dims=4)

    # vertices should be : (last, first, 2nd, 3rd,..., last, X,X,X...)
    # attach the last
    vertices_sorted = tf.concat(
        [tf.expand_dims(vertices_sorted_descending[:,:,:,:,0,:], axis=4), vertices_sorted_ascending], axis=-2) # (b,gr,gc,pr,25,2)
    mask_sorted = tf.concat(
        [tf.expand_dims(mask_sorted_descending[:,:,:,:,0], axis=-1), mask_sorted_ascending], axis=-1) #(b,gr,gc,pr,25)
    return vertices_sorted, tf.constant(mask_sorted)

def calculate_area(vertices_sorted:tf.Tensor, mask_sorted:tf.Tensor):
    '''
    calculate area of intersection
    Args:
        vertices_sorted : (b,gr,gc,pr,25,2)
        mask_sorted : (b,gr,gc,pr,25)
    '''
    vertices_sorted = vertices_sorted * tf.expand_dims(mask_sorted, axis=-1)
    total = vertices_sorted[:,:,:,:, 0:-1, 0] * vertices_sorted[:,:,:,:, 1:,1] - vertices_sorted[:,:,:,:, 0:-1, 1] * vertices_sorted[:,:,:,:, 1:, 0]
    # total (b,gr,gc,24)
    # total = total * mask_sorted[..., :-1]
    total = tf.reduce_sum(total, axis=-1) #(b,gr,gc,pr)
    area = tf.abs(total) / 2.
    return area

def oriented_box_intersection_2d(corners1:tf.Tensor, corners2:tf.Tensor):
    '''
    calculate intersection of 2d rotated rectangles
    Args:
        corners1 : (b,gr,gc,pr,4,2)
        corners2 : (b,gr,gc,pr,4,2)
    Returns:
        area : (b,gr,gc,pr) area of intersection
    '''
    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    vertices_sorted, mask_sorted = sort_indices(vertices, mask)
    area = calculate_area(vertices_sorted, mask_sorted)
    return area

def cal_iou_3d(box3d1:tf.Tensor, box3d2:tf.Tensor, lhw1:tf.Tensor, lhw2:tf.Tensor):
    '''
    calculated 3d iou. assume the 3d bounding boxes are only rotated around z axis
    Args:
        box3d1 : (b,gr,gc,pred, 8,3) (x1y1z1, x1y1z2, x2y1z1, x2y1z2, // x1y2z1, x1y2z2, x2y2z1, x2y1z2)
        box3d2 : (b,gr,gc,pred, 8,3)
    '''
    lhw1 = tf.constant(lhw1)
    lhw2 = tf.constant(lhw2)
    zmax1 = tf.reduce_max(box3d1[..., 2], axis=-1) #(b,gr,gc,pred,)
    zmin1 = tf.reduce_min(box3d1[..., 2], axis=-1)
    zmax2 = tf.reduce_max(box3d2[..., 2], axis=-1)  # (b,gr,gc,pred,)
    zmin2 = tf.reduce_min(box3d2[..., 2], axis=-1)
    z_overlap = tf.maximum(tf.minimum(zmax1, zmax2) - tf.maximum(zmin1, zmin2), 0.)
    # print(z_overlap.shape)
    # print(box3d1[..., 0].shape)
    # print (box3d1[..., 0][:4].shape)
    box1 = tf.stack([box3d1[..., 0][..., :4], box3d1[..., 2][..., :4]], axis=-1) # x,z (b,gr,gc,pred,4,2)
    box2 = tf.stack([box3d2[..., 0][..., :4], box3d2[..., 2][..., :4]], axis=-1)
    # print(box1.shape)
    # print(lhw1.shape)
    inter_2d = oriented_box_intersection_2d(box1, box2)
    # print(inter_2d.shape)
    # print(z_overlap.shape)
    inter_3d = inter_2d * z_overlap
    volume1 = lhw1[...,0] * lhw1[...,1] * lhw1[...,2]
    volume2 = lhw2[..., 0] * lhw2[..., 1] * lhw2[..., 2]
    union_3d = volume1 + volume2 - inter_3d
    # print(tf.reduce_min(union_3d))
    # print('')
    # print(union_3d.shape)
    # print(inter_3d.shape)
    IoU3D = inter_3d / (union_3d + 1e-9)
    # print(tf.reduce_max(inter_2d))
    # print(tf.reduce_max(lhw1[...,0] * lhw1[...,2]))
    # print(tf.reduce_min(inter_2d))
    # print(tf.reduce_max(z_overlap))
    # print(tf.reduce_min(z_overlap))
    # IoU3D = tf.where(IoU3D>1., 1., IoU3D)
    IoU3D_mask = tf.where(union_3d<0., 0., 1.) * tf.where(IoU3D>1., 0., 1.)
    return IoU3D * IoU3D_mask

# import numpy as np
# import time, sys, os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# m = np.reshape(np.array([i for i in range(18)]), (3,3,2))
# a = tf.convert_to_tensor(m)
# # a = a[:,[1,2,0],:]
# a = tf.gather(a, np.reshape([1,2,0]*3, (3,3)), batch_dims=1)
# b = tf.cast(tf.convert_to_tensor(np.ones((3,3))), tf.int32)
# # c = a+b
# # c = tf.gather(a, b)
# print(a)
# c = tf.gather(params=a, indices=b, batch_dims=1)
# c = tf.maximum(a,a)
#
# print(c)









