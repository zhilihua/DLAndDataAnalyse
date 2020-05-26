from __future__ import division
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from DeepLearning.CV.ObjectDetection.utils.CustomLayer.AnchorBoxes import AnchorBoxes
from DeepLearning.CV.ObjectDetection.utils.CustomLayer.L2Normalization import L2Normalization
from DeepLearning.CV.ObjectDetection.utils.CustomLayer.DecodeDetections import DecodeDetections

def ssd_300(image_size,
            n_classes,
            offsets,
            scales,
            mode='training',
            l2_regularization=0.0005,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400):
    '''
    构建一个基于Keras的SSD300模型
    此函数采用的大多数参数仅在锚框层中才需要。如果您正在训练网络，则此处传递的参数
    必须与用于设置“ SSDBoxEncoder”的参数相同。如果您要加载经过训练的权重，则此处
    传递的参数必须与用于产生经过训练的权重的参数相同。
    参数:
        image_size (tuple): 输入的尺寸格式为`(height, width, channels)`.
        n_classes (int): 正样本的类型格式。
        mode (str, optional): 'training', 'inference' 和 'inference_fast'中的一种. 在训练模式下,
            模型输出原始预测张量, 当在推理和快速推理模式下,原始预测被解码为完全坐标和
            通过置信阈值、非最大值抑制、top-k过滤。后面两中模式的不同是：
            “推理”遵循原始Caffe实现的确切过程，而“快速推理”使用更快的预测解码过程。
        l2_regularization (float, optional): L2正则化的值. 应用于所有的卷积层.设置为0则不进行正则化。
        min_scale (float, optional): 锚框大小的最小比例因子，是输入图像的较短边的一部分。
        max_scale (float, optional): 锚框大小的最大缩放比例，占输入图像较短边的比例。 最小和最大之间的所有
        比例因子将被线性插值。 请注意，线性内插缩放因子的倒数第二个实际上是最后一个预测层的缩放因子,
        如果“ two_boxes_for_ar1”为“ True”，则将最后一个缩放因子用于最后一个预测层中纵横比1的第二个框。
        scales (list, optional): 每个卷积预测层包含浮标的浮点列表。此列表必须比预测层数长一个元素。
        如果`two_boxes_for_ar1`是'True'，则前k个元素是k个预测变量层的缩放因子，
        而最后一个元素用于最后一个预测层中纵横比1的第二个框。 即使未使用此附加的最后缩放比例，
        也必须以任何一种方式传递。 如果传递了列表，则此参数将覆盖`min_scale`和`max_scale`。 所有比例因子必须大于零。
        aspect_ratios_global (list, optional): 要为其生成锚框的纵横比列表。 该列表对所有预测层均有效。
        aspect_ratios_per_layer (list, optional): 包含每个预测层的一个长宽比列表的列表。
            这使您可以分别为每个预测层设置纵横比，原始SSD300就是这种情况。 如果传递了一个列表，它将覆盖`aspect_ratios_global`。
        two_boxes_for_ar1 (bool, optional): 仅与包含1的宽高比列表相关。 否则将被忽略。如果为True，将为宽高比1生成两个锚点框。
         第一个将使用相应层的缩放系数生成，第二个将使用所述缩放系数的几何平均值和下一个更大的缩放系数生成。
        steps (list, optional): “None”或具有与预测层相同数量的元素的列表。 元素可以是整数/浮点数，也可以是两个整数/浮点数的元组。
          这些数字代表每个预测层，锚框中心点应沿着图像上的空间网格垂直和水平隔开多少像素。
          如果列表包含整数/浮点数，则该值将用于两个空间尺寸。如果列表包含两个int / float的元组，则它们表示`（step_height，step_width）`。
          如果未提供任何步骤，则将对它们进行计算，以使锚框中心点将在图像尺寸内形成等距的网格。
        offsets (list, optional): “None”或具有与预测层相同数量的元素的列表。 元素可以是浮点数，也可以是两个浮点数的元组。
        这些数字代表每个预测变量层，距图像的顶部和左侧边界多少个像素，最顶部和最左侧的锚框中心点应为“步长”的一部分。
        最后一点很重要：偏移量不是绝对像素值，而是`steps'参数中指定的步长的分数。 如果列表包含浮点数，则该值将用于两个空间尺寸。
        如果列表包含两个浮点数的元组，则它们表示`（vertical_offset，horizontal_offset）`。 如果没有提供偏移，则它们将默认为步长的0.5。
        clip_boxes (bool, optional): 如果为True，则剪切锚框坐标以保持在图像边界内。
        variances (list, optional): 4个浮点数> 0的列表。 每个坐标的锚框偏移将除以其各自的方差值。
        coords (str, optional): 模型在内部使用的盒坐标格式（即，这不是地面真相标签的输入格式）。
        可以是格式（（cx，cy，w，h））（框中心坐标，宽度和高度）的'质心'，格式'（xmin，xmax，ymin，ymax）'的'minmax'或
        格式为（xmin，ymin，xmax，ymax）的'corners'。
        normalize_coords (bool, optional): 如果模型应该使用相对坐标而不是绝对坐标，即模型预测[0,1]内的框坐标而不是绝对坐标，则设置为True。
        subtract_mean (array-like, optional): 广播或与图像形状兼容的任何形状的“无”或整数或浮点值的类似数组的对象。
        该阵列的元素将从图像像素强度值中减去。 例如，传递三个整数的列表以对彩色图像执行每通道平均归一化。
        divide_by_stddev (array-like, optional): “无”或非零整数或与图像形状广播兼容的任何形状的浮点值的类似数组的对象。
         图像像素强度值将被该数组的元素除。 例如，传递三个整数的列表以对彩色图像执行每通道标准偏差归一化。
        swap_channels (list, optional): False或代表要交换输入图像通道的所需顺序的整数列表。
        confidence_thresh (float, optional): 浮点数为[0,1），它是特定肯定类别中的最小分类置信度，以便为各个类别的非最大抑制阶段考虑。
        较低的值将导致选择过程的大部分由非最大抑制阶段完成，而较大的值将导致选择过程的较大部分发生在置信度阈值阶段。
        iou_threshold (float, optional): [0,1]中的浮点数。 Jaccard相似度大于“ iou_threshold”且与局部最大框相似的所有框
        都将从给定类别的预测集中删除，其中“最大”是指框的置信度得分。
        top_k (int, optional): 在非最大抑制阶段之后，每个批次项目将保留的最高评分预测数。
        nms_max_output_size (int, optional): NMS阶段之后剩余的最大预测数。
        return_predictor_sizes (bool, optional): 如果为True，则此函数不仅返回模型，还返回包含预测层空间尺寸的列表。
        由于您始终可以通过Keras API轻松获得它们的大小，因此这并不是绝对必要的，但是以这种方式获得它们很方便且不易出错。
        无论如何，它们仅与训练有关（SSDBoxEncoder需要知道预测层的空间尺寸），因为推断您不需要它们。
    Returns:
        models: Keras的SSD300模型。
        predictor_sizes (optional): 每个卷积预测层包含输出张量形状的（（height，width））部分的Numpy数组。
        在训练期间，生成器功能需要此功能，以便将地面真值标签转换为与模型的输出张量具有相同结构的张量，
        而成本函数又需要该张量。
    '''

    n_predictor_layers = 6 # 对于原始SSD300，网络中的预测转换层数为6。
    n_classes += 1 # 加上背景类的类别总数。
    l2_reg = l2_regularization
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # 排除一些异常。
    ############################################################################
    variances = np.array(variances)
    ############################################################################
    # 计算锚点包围框的参数
    ############################################################################

    # 设置每个预测变量层的纵横比。 这些仅是锚框层所需的。
    aspect_ratios = aspect_ratios_per_layer

    # 计算每个预测层的每个单元格要预测的框数。
    # 我们需要它，以便我们知道预测器层需要具有多少个通道。
    n_boxes = []
    for ar in aspect_ratios_per_layer:
        if (1 in ar) & two_boxes_for_ar1:
            n_boxes.append(len(ar) + 1) # +1 对于第二个宽高比为1的框
        else:
            n_boxes.append(len(ar))

    ############################################################################
    # 在下面定义Lambda图层的功能。
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):   #根据加载的预训练的数据预处理进行
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):   #进行BRG转RGB
        if len(swap_channels) == 3:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # 构建网络
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # 仅需要以下标识层，以便后续的lambda层可以是可选的。
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)

    # 喂conv4_3 到 L2 normalization 层
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    ### 在基础网络之上构建卷积预测层

    # 对每一个包围框我们预测`n_class`个置信值, 因此置信预测深度为`n_boxes * n_classes`， 置信层输出形式为：`(batch, height, width, n_boxes * n_classes)`
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
    # 对每一个包围框我们预测4个坐标, 因此局部预测深度为`n_boxes * 4`， 局部层输出形式为：`(batch, height, width, n_boxes * 4)`
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)

    # 锚点输出形式: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                                             variances=variances, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    variances=variances, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                                        variances=variances, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                                        variances=variances, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4],
                                        variances=variances, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5],
                                        variances=variances, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    ### Reshape

    # 重塑类预测，产生形状为（（batch，height * width * n_boxes，n_classes）`的3D张量
    # 我们希望在最后一个轴中隔离的类对它们执行softmax
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # 重塑盒子的预测，得到形状为（（batch，height * width * n_boxes，4））的3D张量。
    # 我们希望在最后一个轴中隔离四个框坐标来计算平滑的L1损失
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # 重塑锚定框张量，生成形状为（（batch，height * width * n_boxes，8））的3D张量。
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    # 轴0（批处理）和轴2（分别为n_classes或4，）对于所有图层预测都是相同的，
    # 因此我们希望沿轴1连接，即每层的盒子数。`mbox_conf`的输出形状：（batch，n_boxes_total，n_classes）
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # `mbox_loc`输出形式: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # `mbox_priorbox`输出形式: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # 盒子坐标预测将按原样进入损失函数，但是对于类预测，我们将首先应用softmax激活层
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # 将类别和框预测以及锚点连接到一个大的预测向量上。`predictions`的输出形状：（batch，n_boxes_total，n_classes + 4 + 8）
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)

    else:
        raise ValueError("`mode` must be one of 'training', 'inference', but received '{}'.".format(mode))

    return model