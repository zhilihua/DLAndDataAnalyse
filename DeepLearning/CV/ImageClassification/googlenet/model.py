from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, \
    concatenate, AvgPool2D, Dense, Flatten, Dropout, Softmax

#########################搭建googleNet网络###############################
#1、Inception模块构建
def Inception(ch1x1, ch3x3red, ch3x3, ch5x5red,
                 ch5x5, pool_proj, inputs, name):
    # ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5,
    # pool_proj分别对应Inception中各个卷积核的个数,
    branch1 = Conv2D(ch1x1, 1, activation='relu', name=name+'ch1x1')(inputs)

    branch2 = Conv2D(ch3x3red, 1, activation='relu', name=name+'ch3x3red')(inputs)
    branch2 = Conv2D(ch3x3, 3, padding='SAME', activation='relu', name=name+'ch3x3')(branch2)

    branch3 = Conv2D(ch5x5red, 1, activation='relu', name=name+'ch5x5red')(inputs)
    branch3 = Conv2D(ch5x5, 5, padding='SAME', activation='relu', name=name+'ch5x5')(branch3)

    branch4 = MaxPool2D(pool_size=3, strides=1, padding='SAME', name=name+'pool')(inputs)
    branch4 = Conv2D(pool_proj, kernel_size=1, activation='relu', name=name+'pool_proj')(branch4)

    outputs = concatenate([branch1, branch2, branch3, branch4])

    return outputs

#2、InceptionAux模块构建
def InceptionAux(num_classes, inputs, name):
    x = AvgPool2D(pool_size=5, strides=3, name=name+'pool')(inputs)
    x = Conv2D(128, 1, activation='relu', name=name+'conv')(x)
    x = Flatten(name=name+'flat')(x)
    x = Dropout(rate=0.5, name=name+'drop1')(x)
    x = Dense(1024, activation='relu', name=name+'dense1')(x)
    x = Dropout(rate=0.5, name=name+'drop2')(x)
    x = Dense(num_classes, name=name+'dense2')(x)
    x = Softmax(name=name+'softmax')(x)
    return x

#3、整体模型构建
def GoogleNet(inputs, class_num=5, aux_logits=False):
    #接受的为（224，224，3）
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(inputs)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
    x = layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
    x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)
    # Inception模块
    x = Inception(64, 96, 128, 16, 32, 32, x, name="inception_3a")
    x = Inception(128, 128, 192, 32, 96, 64, x, name="inception_3b")
    x = MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
    # Inception模块
    x = Inception(192, 96, 208, 16, 48, 64, x, name="inception_4a")
    # 判断是否使用辅助分类器1。训练时使用，测试时去掉。
    if aux_logits:
        aux1 = InceptionAux(class_num, x, name="aux_1")
    # Inception模块
    x = Inception(160, 112, 224, 24, 64, 64, x, name="inception_4b")
    x = Inception(128, 128, 256, 24, 64, 64, x, name="inception_4c")
    x = Inception(112, 144, 288, 32, 64, 64, x, name="inception_4d")
    # 判断是否使用辅助分类器2。训练时使用，测试时去掉。
    if aux_logits:
        aux2 = InceptionAux(class_num, x, name="aux_2")
    # Inception模块
    x = Inception(256, 160, 320, 32, 128, 128, x, name="inception_4e")
    x = MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)
    # Inception模块
    x = Inception(256, 160, 320, 32, 128, 128, x, name="inception_5a")
    x = Inception(384, 192, 384, 48, 128, 128, x, name="inception_5b")
    # 平均池化层
    x = AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)
    # 拉直
    x = Flatten(name="output_flatten")(x)
    x = Dropout(rate=0.4, name="output_dropout")(x)
    x = Dense(class_num, name="output_dense")(x)
    aux3 = Softmax(name="aux_3")(x)
    # 判断是否使用辅助分类器
    if aux_logits:
        model = Model(inputs=inputs, outputs=[aux1, aux2, aux3])
    else:
        model = Model(inputs=inputs, outputs=aux3)
    return model