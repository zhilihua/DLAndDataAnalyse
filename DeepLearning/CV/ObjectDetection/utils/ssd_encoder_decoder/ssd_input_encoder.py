from __future__ import division
import numpy as np

from bounding_box_utils import iou, convert_coordinates
from ssd_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi

class SSDInputEncoder:
    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 steps,
                 offsets,
                 scales,
                 aspect_ratios_per_layer,
                 two_boxes_for_ar1=True,
                 min_scale=0.1,
                 max_scale=0.9,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 matching_type='multi',
                 pos_iou_threshold=0.5,
                 neg_iou_limit=0.3,
                 normalize_coords=True,
                 background_id=0):
        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ##################################################################################
        # 控制异常.
        ##################################################################################
        scales = np.array(scales)
        variances = np.array(variances)
        ##################################################################################
        # 设置计算成员
        ##################################################################################

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1
        self.predictor_sizes = predictor_sizes    #每一个特征层的shape，即长和宽
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales                         #锚点框缩放因素[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
        self.aspect_ratios = aspect_ratios_per_layer   #每一层锚点框生成比例，基于缩放因素
        """
                 [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
        """
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.steps = steps    #每一层缩放比例[8, 16, 32, 64, 100, 300]
        self.offsets = offsets    #锚点框中心点距离左上角的偏移距离

        self.variances = variances
        self.matching_type = matching_type
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.normalize_coords = normalize_coords
        self.background_id = background_id

        self.n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                self.n_boxes.append(len(aspect_ratios) + 1)
            else:
                self.n_boxes.append(len(aspect_ratios))

        ##################################################################################
        # 为每一个预测层计算锚点包围框
        ##################################################################################
        self.boxes_list = [] # 保存每一层的锚点包围框。

        # 下面的列表仅仅保存诊断信息。
        self.wh_list_diag = [] # 每一个预测层的包围框的宽和高
        self.steps_diag = [] # 每个预测层的任意两个框之间的水平和垂直距离
        self.offsets_diag = [] # 每个预测变量层的偏移
        self.centers_diag = [] # 每个预测变量层的锚框中心点都为（（cy，cx）`

        # 遍历所有预测层，并为每个预测层计算锚框。
        for i in range(len(self.predictor_sizes)):
            boxes, center, wh, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i],
                                                                                   next_scale=self.scales[i+1],
                                                                                   this_steps=self.steps[i],
                                                                                   this_offsets=self.offsets[i],
                                                                                   diagnostics=True)
            self.boxes_list.append(boxes)
            self.wh_list_diag.append(wh)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

    def __call__(self, ground_truth_labels, diagnostics=False):
        # 映射以定义哪些索引代表真实情况中的哪个坐标。
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        batch_size = len(ground_truth_labels)

        ##################################################################################
        # 为y_encoded生成模板。
        ##################################################################################

        y_encoded = self.generate_encoding_template(batch_size=batch_size, diagnostics=False)

        ##################################################################################
        # 匹配真实包围框到锚点框
        ##################################################################################
        y_encoded[:, :, self.background_id] = 1 # 所有包围框默认为背景
        class_vectors = np.eye(self.n_classes) # 用one-hot类向量来定义矩阵

        for i in range(batch_size):
            if ground_truth_labels[i].size == 0: continue
            labels = ground_truth_labels[i].astype(np.float)

            # 如果规范框坐标。
            if self.normalize_coords:
                labels[:, [ymin, ymax]] /= self.img_height
                labels[:, [xmin, xmax]] /= self.img_width

            # 也许转换包围框坐标格式。
            labels = convert_coordinates(labels, start_index=xmin, conversion='corners2centroids')

            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], axis=-1)

            similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[i, :, -12:-8])

            # 对于每个真相框，获取与之最匹配的锚框。
            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            y_encoded[i, bipartite_matches, :-8] = labels_one_hot

            similarities[:, bipartite_matches] = 0

            if self.matching_type == 'multi':  #进行多级匹配
                # 获取所有满足的匹配
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_iou_threshold)

                y_encoded[i, matches[1], :-8] = labels_one_hot[matches[0]]

                similarities[:, matches[1]] = 0

            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_iou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        ##################################################################################
        # 将框坐标转换为锚框偏移量。
        ##################################################################################
        y_encoded[:, :, [-12, -11]] -= y_encoded[:, :, [-8, -7]] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
        y_encoded[:, :, [-12, -11]] /= y_encoded[:, :, [-6, -5]] * y_encoded[:, :, [-4, -3]] # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        y_encoded[:, :, [-10, -9]] /= y_encoded[:, :, [-6, -5]] # w(gt) / w(anchor), h(gt) / h(anchor)
        y_encoded[:, :, [-10, -9]] = np.log(y_encoded[:, :, [-10, -9]]) / y_encoded[:, :, [-2, -1]] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)

        if diagnostics:
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:, :, -12:-8] = 0
            return y_encoded, y_matched_anchors
        else:
            return y_encoded   #[batch_sizes, total_boxes, [one_hot_label, [4个值为真实包围框], [4个值为锚点框] ,[4个值为var]]]

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        next_scale,
                                        this_steps,
                                        this_offsets,
                                        diagnostics=False):
        size = min(self.img_height, self.img_width)
        # 计算所有纵横比的框宽和高
        wh_list = []
        for ar in aspect_ratios:
            if (ar == 1):
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    box_height = box_width = np.sqrt(this_scale * next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        # 计算步长尺寸
        step_height = this_steps
        step_width = this_steps

        # 计算偏执
        offset_height = this_offsets
        offset_width = this_offsets

        # 计算锚框中心点的网格。
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # 创建形状的4D张量模板`(feature_map_height, feature_map_width, n_boxes, 4)`
        # 最后一个维度将包含的位置 `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, ymin, xmax, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')

        if diagnostics:
            return boxes_tensor, (cy, cx), wh_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        # 在所有批处理项目中为每个预测变量层平铺锚定框。
        boxes_batch = []
        for boxes in self.boxes_list:
            # 结果将是3D张量`(batch_size, feature_map_height*feature_map_width*n_boxes, 4)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            boxes = np.reshape(boxes, (batch_size, -1, 4))
            boxes_batch.append(boxes)

        # 将所有层的包围框合成一层[boxes_batch, total_boxes, 4]
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances
        #[boxes_batch, total_boxes, N]
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template