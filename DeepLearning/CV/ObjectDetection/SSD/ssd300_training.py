import tensorflow as tf
from math import ceil

from SSD.models.ssd_300 import ssd_300
from LossModel.ssd_loss import SSDLoss

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from DataAugmentation.data_augmentation import DataAugmentation
from DataAugmentation.object_detection_2d_photometric import ConvertTo3Channels

from ParserData.parser import DataGenerator
from DataAugmentation.object_detection_2d_geometric import Resize

SGD = tf.keras.optimizers.SGD
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler
TerminateOnNaN = tf.keras.callbacks.TerminateOnNaN
CSVLogger = tf.keras.callbacks.CSVLogger
##################################################################
#                           配置参数                             #
##################################################################

img_height = 300
img_width = 300
img_channels = 3
mean_color = [123, 117, 104]
swap_channels = [2, 1, 0]
n_classes = 20
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
variances = [0.1, 0.1, 0.2, 0.2]
normalize_coords = True

###################################################################

##################################################################
#                           构造网络                             #
##################################################################

model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

# 2: 加载预训练参数
weights_path = '../../../../preweights/SSDpreWeights/VGG_ILSVRC_16_layers_fc_reduced.h5'
model.load_weights(weights_path, by_name=True)

sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

#加载之前训练的网络
# model_path = 'path/to/trained/models.h5'
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
# K.clear_session()
# models = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'compute_loss': ssd_loss.compute_loss})

train_dataset = DataGenerator(load_images_into_memory=False)
val_dataset = DataGenerator(load_images_into_memory=False)

VOC_2007_images_dir = '../../../../datasets/VOCdevkit/VOC2007/JPEGImages/'
VOC_2012_images_dir = '../../../../datasets/VOCdevkit/VOC2012/JPEGImages/'

VOC_2007_annotations_dir = '../../../../datasets/VOCdevkit/VOC2007/Annotations/'
VOC_2012_annotations_dir = '../../../../datasets/VOCdevkit/VOC2012/Annotations/'

# 设置图片路径
# VOC_2007_train_image_set_filename = '../../../../datasets/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
# VOC_2012_train_image_set_filename = '../../../../datasets/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
VOC_2007_val_image_set_filename = '../../../../datasets/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
VOC_2012_val_image_set_filename = '../../../../datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
VOC_2007_trainval_image_set_filename = '../../../../datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
VOC_2012_trainval_image_set_filename = '../../../../datasets/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
VOC_2007_test_image_set_filename = '../../../../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

classes = ['background',   #数据的类别
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir,
                                     VOC_2012_images_dir],
                        image_set_filenames=[VOC_2007_trainval_image_set_filename,
                                             VOC_2012_trainval_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir,
                                          VOC_2012_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        )

val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                      image_set_filenames=[VOC_2007_test_image_set_filename],
                      annotations_dirs=[VOC_2007_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      )

batch_size = 16

ssd_data_augmentation = DataAugmentation(img_height=img_height,    #进行数据增广
                                            img_width=img_width,
                                            background=mean_color)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder
                                     )

train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

#定义学习率
def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

model_checkpoint = ModelCheckpoint(filepath='ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=3)

csv_logger = CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

initial_epoch = 0
final_epoch = 120
# steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=ceil(train_dataset_size/batch_size),
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)