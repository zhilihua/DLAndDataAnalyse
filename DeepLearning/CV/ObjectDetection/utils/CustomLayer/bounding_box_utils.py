from __future__ import division
import numpy as np

def convert_coordinates(tensor, start_index, conversion, border_pixels='half'):
    '''
    进行数据格式的转换，支持下面的格式:
        1) (xmin, xmax, ymin, ymax) - 'minmax'格式
        2) (xmin, ymin, xmax, ymax) - 'corners'格式
        2) (cx, cy, w, h) - 'centroids'格式
    '''
    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)

    if conversion == 'corners2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind+2]) / 2.0 # Set cx
        tensor1[..., ind+1] = (tensor[..., ind+1] + tensor[..., ind+3]) / 2.0 # Set cy
        tensor1[..., ind+2] = tensor[..., ind+2] - tensor[..., ind] + d # Set w
        tensor1[..., ind+3] = tensor[..., ind+3] - tensor[..., ind+1] + d # Set h
    elif conversion == 'centroids2corners':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind+2] / 2.0 # Set xmin
        tensor1[..., ind+1] = tensor[..., ind+1] - tensor[..., ind+3] / 2.0 # Set ymin
        tensor1[..., ind+2] = tensor[..., ind] + tensor[..., ind+2] / 2.0 # Set xmax
        tensor1[..., ind+3] = tensor[..., ind+1] + tensor[..., ind+3] / 2.0 # Set ymax

    return tensor1

def intersection_area_(boxes1, boxes2,  mode='outer_product'):
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    if mode == 'outer_product':

        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))

        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

        side_lengths = np.maximum(0, max_xy - min_xy)

        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':

        min_xy = np.maximum(boxes1[:, [xmin, ymin]], boxes2[:, [xmin, ymin]])
        max_xy = np.minimum(boxes1[:, [xmax, ymax]], boxes2[:, [xmax, ymax]])

        side_lengths = np.maximum(0, max_xy - min_xy)

        return side_lengths[:, 0] * side_lengths[:, 1]


def iou(boxes1, boxes2, mode='outer_product', border_pixels='half'):
    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    boxes1 = convert_coordinates(boxes1, start_index=0, conversion='centroids2corners')
    boxes2 = convert_coordinates(boxes2, start_index=0, conversion='centroids2corners')

    intersection_areas = intersection_area_(boxes1, boxes2, mode=mode)

    m = boxes1.shape[0]
    n = boxes2.shape[0]

    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    if border_pixels == 'half':
        d = 0
    elif border_pixels == 'include':
        d = 1
    elif border_pixels == 'exclude':
        d = -1

    if mode == 'outer_product':

        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d), axis=1),
            reps=(1, n))
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d), axis=0),
            reps=(m, 1))

    elif mode == 'element-wise':

        boxes1_areas = (boxes1[:, xmax] - boxes1[:, xmin] + d) * (boxes1[:, ymax] - boxes1[:, ymin] + d)
        boxes2_areas = (boxes2[:, xmax] - boxes2[:, xmin] + d) * (boxes2[:, ymax] - boxes2[:, ymin] + d)

    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas