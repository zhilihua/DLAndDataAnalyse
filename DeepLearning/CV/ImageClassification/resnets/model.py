import tensorflow as tf
from tensorflow.keras import layers, Model

def BasicBlock(filter_num, stride=1, inputs=None, name=None):
    conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same', name=name+'_conv1')(inputs)
    bn1 = layers.BatchNormalization(name=name+'_bn1')(conv1)  # BN层
    relu = layers.Activation('relu', name=name+'_relu')(bn1)  # ReLU激活函数

    conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same', name=name+'_conv2')(relu)
    bn2 = layers.BatchNormalization(name=name+'_bn2')(conv2)  # BN层

    if stride != 1:
        downsample = layers.Conv2D(filter_num, (1, 1), strides=stride, name=name+'_downsample')  # 下采样
    else:
        downsample = lambda x: x  # 恒等映射

    identity = downsample(inputs)  # 恒等映射

    output = layers.add([bn2, identity])  # 主路与支路（恒等映射）相加
    output = tf.nn.relu(output, name=name+'_relu1')  # ReLU激活函数

    return output

def ResNet(layer_dims, num_classes=100, inputs=None, name=None):
    out = layers.Conv2D(64, (3, 3), strides=(1, 1), name=name+'_conv')(inputs)
    out = layers.BatchNormalization(name=name+'_bn')(out)
    out = layers.Activation('relu', name=name+'_relu')(out)
    out = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same', name=name+'_pool')(out)

    layer1 = build_resblock(name+'_layer1', out, 64, layer_dims[0])
    layer2 = build_resblock(name+'_layer2', layer1, 128, layer_dims[1], stride=2)
    layer3 = build_resblock(name+'_layer3', layer2, 256, layer_dims[2], stride=2)
    layer4 = build_resblock(name+'_layer4', layer3, 512, layer_dims[3], stride=2)

    # 全局平均池化
    avgpool = layers.GlobalAveragePooling2D(name=name+'_avgpool')(layer4)
    # 全连接层
    fc = layers.Dense(num_classes, name=name+'_fc')(avgpool)

    model = Model(inputs=inputs, outputs=fc)
    return model

# 构建残差块（将几个相同的残差模块堆叠在一起）
def build_resblock(name, inputs, filter_num, blocks, stride=1):
    # 可能会进行下采样
    out = BasicBlock(filter_num, stride, inputs, name)
    for i in range(1, blocks):
        out = BasicBlock(filter_num, stride=1, inputs=out, name=name+'_%s' %i)
    return out

def resnet18(nums, name='resnet18', inputs=None):
    return ResNet([2, 2, 2, 2], nums, inputs, name)

def resnet34(nums, name='resnet34', inputs=None):
    return ResNet([3, 4, 6, 3], nums, inputs, name)