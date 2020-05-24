from tensorflow.keras import Input, optimizers
from DeepLearning.CV.tools.ClassAug.readDir import dataGenerator, PreDataForCaffe
import math
import os
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Model
import tensorflow as tf

vgg16 = tf.keras.applications.VGG16

def get_nb_files(dirs):
    if not os.path.exists(dirs):
        return 0
    cnt = 0
    for r, dirs, files in os .walk(dirs):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr+'/*')))

    return cnt
#训练模型
def train():
    epochs = 300
    n_classes = 5
    batch_train = 32
    batch_test = 20
    image_w = 224
    image_h = 224
    # 导入数据
    path = '../../../../datasets/traindatas/flower_data/train'  # 读入数据的路径
    path_t = '../../../../datasets/traindatas/flower_data/val'

    Tcnt = get_nb_files(path)
    Vcnt = get_nb_files(path_t)
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
        preprocessing_function=PreDataForCaffe)  # 配置参数

    train_datagen = dataGenerator(**params)
    test_datagen = dataGenerator(preprocessing_function=PreDataForCaffe)

    train_generator = train_datagen.flow_from_directory(path,
                                                        batch_size=batch_train,
                                                        target_size=(image_w, image_h),
                                                        shuffle=True)

    test_generator = test_datagen.flow_from_directory(path_t,
                                                      batch_size=batch_test,
                                                      target_size=(image_w, image_h),
                                                      shuffle=False)
    ############################################################################

    inputs = Input(shape=[image_w, image_h, 3])
    base_model = vgg16(include_top=False,
                       weights='../../../../preweights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                       input_tensor=inputs)

    fla = Flatten()(base_model.output)
    # x = Dense(1024, activation='relu')(fla)
    # x = Dropout(rate=0.5)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(rate=0.5)(x)
    x = Dense(n_classes, activation='softmax')(fla)
    model = Model(inputs=inputs, outputs=x)  # 完成Model构建

    for l in base_model.layers:
        l.trainable = False
    model.summary()

    # 编译模型
    optimizer = optimizers.Adam(lr=0.0005)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['acc'])

    #设置callback
    callbacks = [
        # EarlyStopping(patience=5, verbose=1),
        ModelCheckpoint('saveWeights/vgg11TrsModel.h5', monitor='val_acc', verbose=1,
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
                        steps_per_epoch=math.ceil(Tcnt/batch_train),
                        epochs=epochs,
                        validation_data=test_generator,
                        validation_steps=math.ceil(Vcnt/batch_test),
                        verbose=2,
                        callbacks=callbacks)


if __name__ == '__main__':
    #{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    train()