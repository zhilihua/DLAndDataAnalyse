from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from model import UNet16
from losses import bce_dice_loss, dice_coeff
import data_generater

batch_train = 8
batch_val = 4
height = 256
width = 320

pathModel = "../../../../preweights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"    #预训练数据的参数
path = "../../../../datasets/traindatas/cars/train"   #样本数据集
path_mask = "../../../../datasets/traindatas/cars/train_masks"   #样本mask

path_val = "../../../../datasets/traindatas/cars/val"
path_val_masks = "../../../../datasets/traindatas/cars/val_masks"

def train():
    input_img = Input((height, width, 3), name='img')
    model = UNet16(input_img, weight_path=pathModel)
    model.summary()

    model.compile(optimizer=Adam(lr=1e-4), loss=bce_dice_loss, metrics=[dice_coeff])

    callbacks = [
        EarlyStopping(patience=2, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint('model_seg.h5', verbose=1, save_best_only=True, mode='auto',
                        save_weights_only=False)
    ]

    #生成数据
    train_dataset = data_generater.augment(True, path, path_mask, batch_train, True,
                                           data_generater.PreDataForCaffe,
                                           img_width=width, img_height=height)

    val_dataset = data_generater.augment(False, path, path_mask, batch_train, False,
                                           data_generater.PreDataForCaffe,
                                           img_width=width, img_height=height)

    #训练
    model.fit_generator(train_dataset,
                        steps_per_epoch=200,
                        epochs=20,
                        callbacks=callbacks,
                        validation_data=val_dataset,
                        validation_steps=5,
                        # verbose=2
                        )

if __name__ == '__main__':
    train()