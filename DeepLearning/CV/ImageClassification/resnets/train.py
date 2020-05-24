from tensorflow.keras import Input, optimizers
from DeepLearning.CV.tools.ClassAug.readDir import dataGenerator, PreDataForTF
import math
import os
import glob
from DeepLearning.CV.ImageClassification.resnets.model import resnet18
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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
    batch_train = 24
    batch_test = 10
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
        preprocessing_function=PreDataForTF)  # 配置参数

    train_datagen = dataGenerator(**params)
    test_datagen = dataGenerator(preprocessing_function=PreDataForTF)

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
    model = resnet18(n_classes, inputs=inputs)
    model.summary()

    # 编译模型
    optimizer = optimizers.Adam(lr=0.0003)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['acc'])

    #设置callback
    callbacks = [
        #EarlyStopping(patience=5, verbose=1),
        ModelCheckpoint('saveWeights/resnet18Model.h5', monitor='val_acc', verbose=1,
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
    print("开始resnet训练")
    #{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    train()