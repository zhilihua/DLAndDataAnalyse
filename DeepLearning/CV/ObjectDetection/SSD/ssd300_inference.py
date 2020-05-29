from tensorflow.python.keras.preprocessing import image
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from SSD.models.ssd_300 import ssd_300
import os

img_height = 300
img_width = 300


model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

weights_path = 'ssd300_pascal_07+12_epoch-99_loss-4.0207_val_loss-4.1440.h5'

model.load_weights(weights_path, by_name=True)

paths = '../examples/'
# img_path = '../examples/fish_bike.jpg'
# img_path = '../examples//hogman.jpg'
for maindir, subdir, files in os.walk(paths):

    for file in files:
        orig_images = []
        input_images = []
        img_path = maindir + file

        orig_images.append(imread(img_path))
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img = image.img_to_array(img)
        input_images.append(img)
        input_images = np.array(input_images)

        y_pred = model.predict(input_images)
        confidence_threshold = 0.5

        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        classes = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

        plt.figure(figsize=(20, 12))
        plt.imshow(orig_images[0])

        current_axis = plt.gca()

        for box in y_pred_thresh[0]:
            xmin = box[2] * orig_images[0].shape[1] / img_width
            ymin = box[3] * orig_images[0].shape[0] / img_height
            xmax = box[4] * orig_images[0].shape[1] / img_width
            ymax = box[5] * orig_images[0].shape[0] / img_height
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

        plt.show()