from tensorflow.keras import Input, optimizers
from DeepLearning.CV.tools.ClassAug.readDir import dataGenerator, PreDataForVGG
import math
import os
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf

googlenet = tf.keras.applications.InceptionV3

#训练模型
def train():
    epochs = 300
    n_classes = 5
    batch_train = 64
    batch_test = 20
    image_w = 224
    image_h = 224
    # 导入数据
    path = '../../../../datasets/traindatas/flower_data/train'  # 读入数据的路径
    path_t = '../../../../datasets/traindatas/flower_data/val'

    ################################自写数据生成方法###############################
    '''
    datagen = dataGenerator()  # 实例化数据生成器
    # 执行一次获取数据
    datagen.readData(path)
    samples_num = datagen.num_samples

    train_datas = datagen.augment(True, path, batch_train, True, PreDataForVGG, image_w, image_h)  # 导入数据并进行增强
    datagen_t = dataGenerator()
    # 执行一次获取数据
    datagen_t.readData(path_t)
    samples_num_t = datagen_t.num_samples

    val_datas = datagen_t.augment(False, path_t, batch_test, False, PreDataForVGG, image_w, image_h)
    '''
    ############################################################################
    params = dict(  # rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=PreDataForVGG)  # 配置参数

    train_datagen = dataGenerator(**params)
    test_datagen = dataGenerator(preprocessing_function=PreDataForVGG)

    train_generator = train_datagen.flow_from_directory(path,
                                                        batch_size=batch_train,
                                                        target_size=(image_w, image_h),
                                                        shuffle=True)

    test_generator = test_datagen.flow_from_directory(path_t,
                                                      batch_size=batch_test,
                                                      target_size=(image_w, image_h),
                                                      shuffle=False)

    train_num = train_generator.samples
    test_num = test_generator.samples

    ############################################################################

    inputs = Input(shape=[image_w, image_h, 3])
    base_model = googlenet(include_top=False,
                       weights='../../../../preweights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                       input_tensor=inputs)

    # fla = Flatten()(base_model.output)
    fla = GlobalAveragePooling2D()(base_model.output)
    # x = Dropout(rate=0.5)(fla)
    x = Dense(1024, activation='relu')(fla)
    x = Dropout(rate=0.5)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    output = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)  # 完成Model构建

    base_model.trainable = False

    model.summary()

    # 编译模型
    optimizer = optimizers.Adam(lr=0.0003)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['acc'])

    #设置callback
    callbacks = [
        # EarlyStopping(patience=5, verbose=1),
        ModelCheckpoint('saveWeights/googlenetTrsModel.h5', monitor='val_acc', verbose=1,
                        save_best_only=True, mode='max')
    ]
    # 训练
    '''
    model.fit_generator(train_datas,
                        steps_per_epoch=math.ceil(samples_num/batch_train),
                        epochs=epochs,
                        validation_data=val_datas,
                        validation_steps=math.ceil(samples_num_t/batch_test),
                        verbose=2,
                        callbacks=callbacks)'''
    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(train_num/batch_train),
                        epochs=epochs,
                        validation_data=test_generator,
                        validation_steps=math.ceil(test_num/batch_test),
                        verbose=2,
                        callbacks=callbacks)


if __name__ == '__main__':
    #{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    train()