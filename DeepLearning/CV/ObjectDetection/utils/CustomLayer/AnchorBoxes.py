from __future__ import division
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer

from DeepLearning.CV.ObjectDetection.utils.CustomLayer.bounding_box_utils import convert_coordinates

class AnchorBoxes(Layer):
    '''
    基于输入张量和传递的参数，创建包含锚框坐标和方差的输出张量的Keras层。
    输入张量的每一个空间单元创建一系列的不同比例的2D锚点包围框。依据`aspect_ratios` 和 `two_boxes_for_ar1`
    参数决定锚点包围框被创建的个数，默认情况下为4.这些框由坐标元组参数化(xmin, xmax, ymin, ymax)。
    在网络中这层的目的是在推理过程中模型能够自给自足。因为模型预测的是包围框的偏执（不是直接预测坐标），所以为了
    从预测偏执中构造最终的预测，这一点是必须的。如果模型的输出张量不包含锚框坐标，则模型输出中将缺少将预测的偏移
    量转换回绝对坐标的必要信息。这就是必须预测锚点包围框，而不是预测完全坐标的解释。
    输入:
        4D 张量 `(batch, height, width, channels)`
    输出:
        5D 张量 `(batch, height, width, n_boxes, 8)`.
        最后一维包含4个锚点坐标和每个包围框的4个方差值。
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 next_scale,
                 this_steps,
                 this_offsets,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 normalize_coords=False,
                 **kwargs):
        '''
        所有参数的值必须设置为跟包围框编码过程一样的值。否则，行为不能被定义。一些参数将会在SSDBoxEncoder中进行详细的解释。
        Arguments:
            img_height (int): 输入图像的高度（height）。
            img_width (int): 输入图像的高度（width）。
            this_scale (float): 范围在[0, 1]中的浮点型，生成的锚框大小的比例因子，即输入图像的较短边的一部分。
            next_scale (float): 范围在[0, 1]中的浮点型, 下一个比较大的比例因子，仅仅在`self.two_boxes_for_ar1 == True`
            时起作用。
            aspect_ratios (list, optional): 要为此层生成默认框的纵横比列表。
            two_boxes_for_ar1 (bool, optional): 仅仅和参数`aspect_ratios`相关，包含1。
                如果为`True`，默认框将会按照比例为1生成默认框。首先各自层用比例因子生成包围框，然后用比例因子和
                接下来更大的比例因子的几何均值生成。
            clip_boxes (bool, optional): 如果为`True`,剪切锚框坐标以保留在图像边界内。
            variances (list, optional): 一个4维的大于0的浮点型列表。 每个坐标的锚框偏移将除以其各自的方差值。
            coords (str, optional): 在模型中被用于内部的坐标格式。
            normalize_coords (bool, optional): 如果模型使用相对坐标而不是绝对坐标设置为`True.
        '''
        variances = np.array(variances)

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.variances = variances
        self.normalize_coords = normalize_coords
        # 计算没有个单元的包围框个数
        if (1 in aspect_ratios) and two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        基于输入张量的shape，返回一个锚点框张量。
        实现逻辑和模块`ssd_box_encode_decode_utils.py`一致。
        请注意，该张量在运行时不参与任何图形计算。 在图形创建期间将其一次创建为一个常数，
        并在运行时将其与模型输出的其余部分一起输出。 因此，所有逻辑都实现为Numpy数组操作，
        并且在将其输出之前将最终的Numpy数组转换为Keras张量就足够了。
        Arguments:
            x (tensor): 4维张量 `(batch, height, width, channels)` .
            这一层的输入必须维标准化预测层的输出。
        '''
        #计算每一个基于纵横比的包围框的宽和高，图像较短的边将用于计算w和h，用`scale`和`aspect_ratios`。
        size = min(self.img_height, self.img_width)
        # 计算所有纵横比例包围框的宽和高。
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1):
                # 计算纵横比为1的标准锚点包围框。
                box_height = box_width = self.this_scale * size
                wh_list.append((box_width, box_height))
                if self.two_boxes_for_ar1:
                    # 使用此比例值和下一个比例的几何平均值计算一个稍大的版本。
                    box_height = box_width = np.sqrt(self.this_scale * self.next_scale) * size
                    wh_list.append((box_width, box_height))
            else:
                box_height = self.this_scale * size / np.sqrt(ar)
                box_width = self.this_scale * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        # 我们需要输入张量的尺寸
        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.shape

        # 计算中心点的格子线，它们对于所有宽高比都是相同的。
        # 计算步长，即锚框中心点在垂直和水平方向上相距多远。
        step_height = self.this_steps
        step_width = self.this_steps

        # 计算偏移量，即第一个锚点框中心点的像素值是从图像的顶部和左侧开始。
        offset_height = self.this_offsets
        offset_width = self.this_offsets

        # 现在我们有了偏移量和步长，计算锚点盒中心点的网格。
        cy = np.linspace((offset_height * step_height), (offset_height + int(feature_map_height) - 1) * step_height, int(feature_map_height))
        cx = np.linspace(offset_width * step_width, (offset_width + int(feature_map_width) - 1) * step_width, int(feature_map_width))
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # 这对于np.tile（）进行进一步的操作是必要的
        cy_grid = np.expand_dims(cy_grid, -1) # 这对于np.tile（）进行进一步的操作是必要的

        # 创造一个4维张量尺寸模板`(feature_map_height, feature_map_width, n_boxes, 4)`
        # 最后一个维度将包含`（cx，cy，w，h）`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # 设置 cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # 设置 cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # 设置 w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # 设置 h

        # 转换 `(cx, cy, w, h)` 为 `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

        # 如果启用了“ normalize_coords”，则将坐标标准化为[0,1]以内
        if self.normalize_coords:
            boxes_tensor[:, :, :, [0, 2]] /= self.img_width
            boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        # 转换 `(xmin, ymin, xmax, ymax)` 为 `(cx, cy, w, h)`.
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids', border_pixels='half')


        # 创建一个张量以包含方差并将其附加到`boxes_tensor`。 该张量具有与“ boxes_tensor”相同的形状，
        # 并且对于最后一个轴上的每个位置仅包含相同的4个方差值。
        variances_tensor = np.zeros_like(boxes_tensor) # 形状为 `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor += self.variances
        # 现在 `boxes_tensor` 变为一个形状为 `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # 现在在“ boxes_tensor”前面添加一个尺寸以说明批量大小并将其平铺
        # 结果将为一个5维的数据`(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'variances': list(self.variances),
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))