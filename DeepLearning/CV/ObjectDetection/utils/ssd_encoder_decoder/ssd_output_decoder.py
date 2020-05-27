from __future__ import division
import numpy as np

from bounding_box_utils import iou, convert_coordinates

def _greedy_nms(predictions, iou_threshold=0.45,
                #coords='corners',
                border_pixels='half'):
    #通过非最大抑制，筛选包围框。
    boxes_left = np.copy(predictions)
    maxima = [] # 这是我们存储通过非最大抑制使的盒子
    while boxes_left.shape[0] > 0:
        maximum_index = np.argmax(boxes_left[:, 0]) # ...获取具有最高置信度的下一个框的索引...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...复制此包围框...
        maxima.append(maximum_box)
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # 从`boxes_left`移除这个包围框
        if boxes_left.shape[0] == 0: break
        similarities = iou(boxes_left[:, 1:], maximum_box[1:],
                           #coords=coords,
                           mode='element-wise', border_pixels=border_pixels)
        boxes_left = boxes_left[similarities <= iou_threshold]
    return np.array(maxima)

def decode_detections(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, "
                         "the decoder needs the image size in order to decode the predictions, but `img_height == {}` "
                         "and `img_width == {}`".format(img_height, img_width))

    # 1: 将框坐标从预测的锚框偏移量转换为预测的绝对坐标

    y_pred_decoded_raw = np.copy(y_pred[:, :, :-8]) # shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: 如果模型应用了归一化，那么执行下面操作。

    if normalize_coords:
        y_pred_decoded_raw[:, :, [-4, -2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:, :, [-3, -1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: 对每个类别应用置信度阈值和非最大抑制

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # 类的数量

    y_pred_decoded = [] # 将最终预测存储在此列表中
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # 在此存储此批次项目的最终预测
        for class_id in range(1, n_classes):
            single_class = batch_item[:, [class_id, -4, -3, -2, -1]]
            threshold_met = single_class[single_class[:, 0] > confidence_thresh]
            if threshold_met.shape[0] > 0: # 如果有任何盒子达到阈值...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold,
                                     #coords='corners',
                                     border_pixels=border_pixels) # ...执行NMS。
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id
                maxima_output[:,1:] = maxima
                pred.append(maxima_output)
        # 完成所有类别后，仅保留得分最高的`top_k`最大值
        if pred:
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # 如果目前尚余`top_k`个结果，则没有任何可过滤的内容,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...获取“ top_k”最高得分最大值的索引...
                pred = pred[top_k_indices]
        else:
            pred = np.array(pred)
        y_pred_decoded.append(pred)

    return y_pred_decoded