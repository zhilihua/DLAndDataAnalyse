from tensorflow.python.keras.preprocessing import image
from imageio import imread
import cv2
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

#随机生成颜色函数
def random_color(num=21):
    RGB=[]
    for i in range(3):
        c = np.random.randint(0, 255, size=num)
        c = [int(i) for i in c]
        RGB.append(c)
    return list(zip(*RGB))


def run(paths, mode, delay=0, size_scale=2):
    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    if mode == 'pic':
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
                    current_axis.add_patch(
                        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
                    current_axis.text(xmin, ymin, label, size='x-large', color='white',
                                      bbox={'facecolor': color, 'alpha': 1.0})

                plt.show()
    elif mode == 'video':
        colors = random_color()

        cap = cv2.VideoCapture(paths)
        size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cv2.namedWindow("frame", 0)
        cv2.resizeWindow('frame', int(size[0]*size_scale), int(size[1])*size_scale)
        while True:
            orig_images = []
            input_images = []
            # 一帧一帧的捕获
            ret, frame = cap.read()

            orig_images.append(frame)
            input_images.append(cv2.resize(frame, (img_height, img_width)))
            input_images = np.array(input_images)
            if ret != True:
                break

            y_pred = model.predict(input_images[..., ::-1])    #opencv读取的是BGR通道顺序，需要进行转换为RGB
            confidence_threshold = 0.5

            y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
            np.set_printoptions(precision=2, suppress=True, linewidth=90)

            for box in y_pred_thresh[0]:
                xmin = box[2] * orig_images[0].shape[1] / img_width
                ymin = box[3] * orig_images[0].shape[0] / img_height
                xmax = box[4] * orig_images[0].shape[1] / img_width
                ymax = box[5] * orig_images[0].shape[0] / img_height

                color = colors[int(box[0])]

                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

                cv2.rectangle(orig_images[0], (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=color, thickness=1)
                cv2.putText(orig_images[0], label, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255), 1)

            cv2.imshow('frame', orig_images[0])
            if (cv2.waitKey(10) & 0xFF) == ord('q'):
                break
            cv2.waitKey(delay)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print('输入模式不对！！！！')

if __name__ == '__main__':
    # paths = '../examples/PIC/'
    paths = '../examples/VIDEO/Hallway_Original_sequence.avi'
    # run(paths, 'pic')
    run(paths, 'video', delay=100)