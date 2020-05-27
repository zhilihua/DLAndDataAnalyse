from __future__ import division
import numpy as np
from math import ceil
from tqdm import trange
import sys
import warnings

from utils.ParserData.parser import DataGenerator
from utils.DataAugmentation.object_detection_2d_geometric import Resize

from utils.DataAugmentation.object_detection_2d_photometric import ConvertTo3Channels
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from utils.DataAugmentation.object_detection_2d_misc_utils import apply_inverse_transforms

from utils.ssd_encoder_decoder.bounding_box_utils import iou

class Evaluator:
    '''
    计算在给定数据集上SSD模型的Map。有选择的返回average precisions, precisions, and recalls.
    '''

    def __init__(self,
                 model,
                 n_classes,
                 data_generator,
                 model_mode='inference',
                 pred_format={'class_id': 0, 'conf': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5},
                 gt_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            model (Keras model): 一个SSD模型对象.
            n_classes (int): 正样本数量,例如Pascal有20个，MS COCO有80个。
            data_generator (DataGenerator): 数据生成器.
            model_mode (str, optional): 模型的模式，训练（training）或者推理模式（inference）。
            pred_format (dict, optional): 预测数据的数据格式。
            gt_format (list, optional): 真实数据的数据格式。
        '''

        if not isinstance(data_generator, DataGenerator):
            warnings.warn("`data_generator` is not a `DataGenerator` object, which will cause undefined behavior.")

        self.model = model
        self.data_generator = data_generator
        self.n_classes = n_classes
        self.model_mode = model_mode
        self.pred_format = pred_format
        self.gt_format = gt_format

        self.prediction_results = None     #预测结果
        self.num_gt_per_class = None
        self.true_positives = None
        self.false_positives = None
        self.cumulative_true_positives = None
        self.cumulative_false_positives = None
        self.cumulative_precisions = None # "Cumulative" 表示每个列表中的第i个元素代表该类别的前i个最高置信度预测的精度。
        self.cumulative_recalls = None # "Cumulative" 是指每个列表中的第i个元素代表该类别的前i个最高置信度预测的召回率。
        self.average_precisions = None
        self.mean_average_precision = None

    def __call__(self,
                 img_height,
                 img_width,
                 batch_size,
                 data_generator_mode='resize',
                 round_confidences=False,
                 matching_iou_threshold=0.5,
                 border_pixels='include',
                 sorting_algorithm='quicksort',
                 average_precision_mode='sample',
                 num_recall_points=11,
                 ignore_neutral_boxes=True,
                 return_precisions=False,
                 return_recalls=False,
                 return_average_precisions=False,
                 verbose=True,
                 decoding_confidence_thresh=0.01,
                 decoding_iou_threshold=0.45,
                 decoding_top_k=200,
                 decoding_pred_coords='centroids',
                 decoding_normalize_coords=True):
        '''
        Arguments:
            img_height (int): 输入图片高度
            img_width (int): 输入图片宽度
            batch_size (int): 验证尺寸
            data_generator_mode (str, optional): 数据生成模式。
            round_confidences (int, optional): 是否进行舍弃小数部分。
                confidences will be rounded to. If `False`, the confidences will not be rounded.
            matching_iou_threshold (float, optional): 如果某个预测的Jaccard与同一类别的任何gt边界框至少有“ matching_iou_threshold”重叠，则该预测将被视为TP。
            border_pixels (str, optional): 如何处理边界框的边框像素。
            sorting_algorithm (str, optional): 在匹配算法中，哪个排序算法被应用。
            average_precision_mode (str, optional): 计算AP模式。
            num_recall_points (int, optional): 要从精度调用曲线中采样以计算平均精度的点数。 换句话说，这是将计算得出的精度的等距召回值的数量。
            ignore_neutral_boxes (bool, optional): 是否忽略中性框。
            return_precisions (bool, optional): If `True`, 返回一个嵌套列表，其中包含每个类的累积精度。
            return_recalls (bool, optional): If `True`, 返回一个嵌套列表，其中包含每个类的累积召回率。
            return_average_precisions (bool, optional): If `True`, 返回包含每个类的AP的列表。
            verbose (bool, optional): If `True`, 将在运行时打印进度。
            decoding_confidence_thresh (float, optional): 解码时候阈值大于该值的框才会被选择。
            decoding_iou_threshold (float, optional): NMS结算的阈值。
            decoding_top_k (int, optional): 保留的前K个框。
            decoding_normalize_coords (bool, optional): 是否进行归一化。
        Returns:
           浮点数，MAP以及参数中指定的所有可选返回值。
        '''

        #############################################################################################
        # 在整个数据集上进行预测。
        #############################################################################################

        self.predict_on_dataset(img_height=img_height,
                                img_width=img_width,
                                batch_size=batch_size,
                                data_generator_mode=data_generator_mode,
                                decoding_confidence_thresh=decoding_confidence_thresh,
                                decoding_iou_threshold=decoding_iou_threshold,
                                decoding_top_k=decoding_top_k,
                                decoding_pred_coords=decoding_pred_coords,
                                decoding_normalize_coords=decoding_normalize_coords,
                                decoding_border_pixels=border_pixels,
                                round_confidences=round_confidences,
                                verbose=verbose,
                                ret=False)

        #############################################################################################
        # 获取每个类的gt框的总数。
        #############################################################################################

        self.get_num_gt_per_class(ignore_neutral_boxes=ignore_neutral_boxes,
                                  verbose=False,
                                  ret=False)

        #############################################################################################
        # 将所有类别的预测与gt框相匹配。
        #############################################################################################

        self.match_predictions(ignore_neutral_boxes=ignore_neutral_boxes,
                               matching_iou_threshold=matching_iou_threshold,
                               border_pixels=border_pixels,
                               sorting_algorithm=sorting_algorithm,
                               verbose=verbose,
                               ret=False)

        #############################################################################################
        # 计算所有类的累积精度和召回率。
        #############################################################################################

        self.compute_precision_recall(verbose=verbose, ret=False)

        #############################################################################################
        # 计算此类的平均精度。
        #############################################################################################

        self.compute_average_precisions(mode=average_precision_mode,
                                        num_recall_points=num_recall_points,
                                        verbose=verbose,
                                        ret=False)

        #############################################################################################
        # 计算MAP。
        #############################################################################################

        mean_average_precision = self.compute_mean_average_precision(ret=True)

        #############################################################################################

        # 编译结果.
        if return_precisions or return_recalls or return_average_precisions:
            ret = [mean_average_precision]
            if return_average_precisions:
                ret.append(self.average_precisions)
            if return_precisions:
                ret.append(self.cumulative_precisions)
            if return_recalls:
                ret.append(self.cumulative_recalls)
            return ret
        else:
            return mean_average_precision

    def predict_on_dataset(self,
                           img_height,
                           img_width,
                           batch_size,
                           data_generator_mode='resize',
                           decoding_confidence_thresh=0.01,
                           decoding_iou_threshold=0.45,
                           decoding_top_k=200,
                           decoding_pred_coords='centroids',
                           decoding_normalize_coords=True,
                           decoding_border_pixels='include',
                           round_confidences=False,
                           verbose=True,
                           ret=False):
        '''
        在`data_generator`给定的整个数据集上运行给定模型的预测。
        Arguments:
            img_height (int): 输入图片高度
            img_width (int): 输入图片宽度
            batch_size (int): 验证尺寸
            data_generator_mode (str, optional): 数据生成模式。
            decoding_confidence_thresh (float, optional): 解码时候阈值大于该值的框才会被选择。
            decoding_iou_threshold (float, optional): NMS结算的阈值。
            decoding_top_k (int, optional): 保留的前K个框。
            decoding_normalize_coords (bool, optional): 是否进行归一化。
            round_confidences (int, optional): 是否对精度进行舍弃。
            verbose (bool, optional): If `True`, 运行时候打印进度。
            ret (bool, optional): If `True`, 返回预测结果。
        Returns:
        '''

        class_id_pred = self.pred_format['class_id']
        conf_pred = self.pred_format['conf']
        xmin_pred = self.pred_format['xmin']
        ymin_pred = self.pred_format['ymin']
        xmax_pred = self.pred_format['xmax']
        ymax_pred = self.pred_format['ymax']

        #############################################################################################
        # 配置验证阶段的数据生成器。
        #############################################################################################

        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height, width=img_width, labels_format=self.gt_format)
        if data_generator_mode == 'resize':   #数据转换方式
            transformations = [convert_to_3_channels,
                               resize]
        else:
            raise ValueError("`data_generator_mode` can be either of 'resize' or 'pad', but received '{}'.".format(data_generator_mode))

        # Set the generator parameters.
        generator = self.data_generator.generate(batch_size=batch_size,
                                                 shuffle=False,
                                                 transformations=transformations,
                                                 label_encoder=None,
                                                 returns={
                                                          'image_ids',
                                                          'inverse_transform',
                                                          },
                                                 # keep_images_without_gt=True,
                                                 # degenerate_box_handling='remove'
                                                 )

        if self.data_generator.image_ids is None:
            self.data_generator.image_ids = list(range(self.data_generator.get_dataset_size()))

        #############################################################################################
        # Predict over all batches of the dataset and store the predictions.
        #############################################################################################

        # We have to generate a separate results list for each class.
        results = [list() for _ in range(self.n_classes + 1)]

        # Compute the number of batches to iterate over the entire dataset.
        n_images = self.data_generator.get_dataset_size()
        n_batches = int(ceil(n_images / batch_size))
        if verbose:
            print("Number of images in the evaluation dataset: {}".format(n_images))
            print()
            tr = trange(n_batches, file=sys.stdout)
            tr.set_description('Producing predictions batch-wise')
        else:
            tr = range(n_batches)

        # 循环所有批次
        for j in tr:
            # Generate batch.
            batch_X, batch_y, batch_image_ids,  batch_inverse_transforms = next(generator)
            # Predict.
            y_pred = self.model.predict(batch_X)
            # If the model was created in 'training' mode, the raw predictions need to
            # be decoded and filtered, otherwise that's already taken care of.
            if self.model_mode == 'training':
                # Decode.
                y_pred = decode_detections(y_pred,
                                           confidence_thresh=decoding_confidence_thresh,
                                           iou_threshold=decoding_iou_threshold,
                                           top_k=decoding_top_k,
                                           input_coords=decoding_pred_coords,
                                           normalize_coords=decoding_normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width,
                                           border_pixels=decoding_border_pixels)
            else:
                # Filter out the all-zeros dummy elements of `y_pred`.
                y_pred_filtered = []
                for i in range(len(y_pred)):
                    y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0])
                y_pred = y_pred_filtered
            # Convert the predicted box coordinates for the original images.
            y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)

            # Iterate over all batch items.
            for k, batch_item in enumerate(y_pred):

                image_id = batch_image_ids[k]

                for box in batch_item:
                    class_id = int(box[class_id_pred])
                    # 舍入框坐标以减少所需的内存（不要小数点后面内容）.
                    if round_confidences:
                        confidence = round(box[conf_pred], round_confidences)
                    else:
                        confidence = box[conf_pred]
                    xmin = round(box[xmin_pred], 1)
                    ymin = round(box[ymin_pred], 1)
                    xmax = round(box[xmax_pred], 1)
                    ymax = round(box[ymax_pred], 1)
                    prediction = (image_id, confidence, xmin, ymin, xmax, ymax)
                    # Append the predicted box to the results list for its class.
                    results[class_id].append(prediction)

        self.prediction_results = results

        if ret:
            return results

    def get_num_gt_per_class(self,
                             ignore_neutral_boxes=True,
                             verbose=True,
                             ret=False):
        '''
        计算数据集中每个类别的gt框的数量。
        Arguments:
            ignore_neutral_boxes (bool, optional): 是否忽略中性框。
            verbose (bool, optional): 如果为True，将在运行时打印进度。
            ret (bool, optional): If `True`,返回计数列表。
        Returns:
            每个类别的gt框的数量
        '''

        if self.data_generator.labels is None:
            raise ValueError("Computing the number of ground truth boxes per class not possible, no ground truth given.")

        num_gt_per_class = np.zeros(shape=(self.n_classes+1), dtype=np.int)

        class_id_index = self.gt_format['class_id']

        ground_truth = self.data_generator.labels    #总共图片的数量

        if verbose:
            print('Computing the number of positive ground truth boxes per class.')
            tr = trange(len(ground_truth), file=sys.stdout)
        else:
            tr = range(len(ground_truth))

        # Iterate over the ground truth for all images in the dataset.
        for i in tr:

            boxes = np.asarray(ground_truth[i])   #单张图片中所有包围框

            # 迭代当前图片的所有gt包围框
            for j in range(boxes.shape[0]):

                if ignore_neutral_boxes and not (self.data_generator.eval_neutral is None):
                    if not self.data_generator.eval_neutral[i][j]:
                        # 如果此框不应该与评估无关，则增加相应类ID的计数器。
                        class_id = boxes[j, class_id_index]
                        num_gt_per_class[class_id] += 1
                else:

                    class_id = boxes[j, class_id_index]
                    num_gt_per_class[class_id] += 1

        self.num_gt_per_class = num_gt_per_class

        if ret:
            return num_gt_per_class

    def match_predictions(self,
                          ignore_neutral_boxes=True,
                          matching_iou_threshold=0.5,
                          border_pixels='include',
                          sorting_algorithm='quicksort',
                          verbose=True,
                          ret=False):
        '''

        使预测与gt框匹配。注意：调用之前，predict_on_dataset()必须被调用。
        Arguments:
            ignore_neutral_boxes (bool, optional): 是否忽略中性框。
            matching_iou_threshold (float, optional): 被认定为TP的阈值。
            border_pixels (str, optional): 如何处理边界框的边框像素。
            sorting_algorithm (str, optional): 匹配算法应使用哪种排序算法。
            verbose (bool, optional): If `True`, 打印运行进度。
            ret (bool, optional): If `True`, 返回TP和FP。
        Returns:
        '''

        if self.data_generator.labels is None:
            raise ValueError("Matching predictions to ground truth boxes not possible, no ground truth given.")

        if self.prediction_results is None:
            raise ValueError("There are no prediction results. You must run `predict_on_dataset()` before calling this method.")

        class_id_gt = self.gt_format['class_id']
        xmin_gt = self.gt_format['xmin']
        ymin_gt = self.gt_format['ymin']
        xmax_gt = self.gt_format['xmax']
        ymax_gt = self.gt_format['ymax']

        # 将gt实况转换为更有效的格式，这是我们需要做的，即通过图像ID重复访问gt实况。
        ground_truth = {}
        eval_neutral_available = not (self.data_generator.eval_neutral is None)
        for i in range(len(self.data_generator.image_ids)):
            image_id = str(self.data_generator.image_ids[i])
            labels = self.data_generator.labels[i]
            if ignore_neutral_boxes and eval_neutral_available:
                ground_truth[image_id] = (np.asarray(labels), np.asarray(self.data_generator.eval_neutral[i]))
            else:
                ground_truth[image_id] = np.asarray(labels)

        true_positives = [[]] # 每个类别的TP，以递减的置信度排序。
        false_positives = [[]] # 每个类别的FP，以递减的置信度排序。
        cumulative_true_positives = [[]]
        cumulative_false_positives = [[]]

        # 迭代所有类。
        for class_id in range(1, self.n_classes + 1):

            predictions = self.prediction_results[class_id]

            # 将匹配结果存储在这些列表中:
            true_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a true positive, 0 otherwise
            false_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a false positive, 0 otherwise

            # 如果这个类别根本没有任何预测，我们就在这里完成。
            if len(predictions) == 0:
                print("No predictions for class {}/{}".format(class_id, self.n_classes))
                true_positives.append(true_pos)
                false_positives.append(false_pos)
                continue

            num_chars_per_image_id = len(str(predictions[0][0])) + 6   #请保留一些字符，以防某些图像ID比其他图像ID长。

            preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_per_image_id)),
                                        ('confidence', 'f4'),
                                        ('xmin', 'f4'),
                                        ('ymin', 'f4'),
                                        ('xmax', 'f4'),
                                        ('ymax', 'f4')])

            predictions = np.array(predictions, dtype=preds_data_type)

            descending_indices = np.argsort(-predictions['confidence'], kind=sorting_algorithm)  #加负号可以式其按降序排列
            predictions_sorted = predictions[descending_indices]  #降序后的预测结果

            if verbose:
                tr = trange(len(predictions), file=sys.stdout)
                tr.set_description("Matching predictions to ground truth, class {}/{}.".format(class_id, self.n_classes))
            else:
                tr = range(len(predictions.shape))

            gt_matched = {}

            # 迭代所有预测。
            for i in tr:

                prediction = predictions_sorted[i]
                image_id = prediction['image_id']
                pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']]))

                if ignore_neutral_boxes and eval_neutral_available:
                    gt, eval_neutral = ground_truth[image_id]
                else:
                    gt = ground_truth[image_id]
                gt = np.asarray(gt)
                class_mask = gt[:, class_id_gt] == class_id  # 算出该类的掩码，方便提取此类框
                gt = gt[class_mask]
                if ignore_neutral_boxes and eval_neutral_available:
                    eval_neutral = eval_neutral[class_mask]

                if gt.size == 0:
                    false_pos[i] = 1
                    continue

                # 用同一类的所有gt框计算此预测的IoU。
                overlaps = iou(boxes1=gt[:,[xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
                               boxes2=pred_box,
                               #coords='corners',
                               mode='element-wise',
                               border_pixels=border_pixels)

                gt_match_index = np.argmax(overlaps)
                gt_match_overlap = overlaps[gt_match_index]

                if gt_match_overlap < matching_iou_threshold:
                    false_pos[i] = 1
                else:
                    if not (ignore_neutral_boxes and eval_neutral_available) or (eval_neutral[gt_match_index] == False):
                        if not (image_id in gt_matched):
                            true_pos[i] = 1
                            gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
                            gt_matched[image_id][gt_match_index] = True
                        elif not gt_matched[image_id][gt_match_index]:
                            true_pos[i] = 1
                            gt_matched[image_id][gt_match_index] = True
                        else:
                            false_pos[i] = 1

            true_positives.append(true_pos)
            false_positives.append(false_pos)

            cumulative_true_pos = np.cumsum(true_pos) # Cumulative sums of the true positives
            cumulative_false_pos = np.cumsum(false_pos) # Cumulative sums of the false positives

            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)

        self.true_positives = true_positives
        self.false_positives = false_positives
        self.cumulative_true_positives = cumulative_true_positives
        self.cumulative_false_positives = cumulative_false_positives

        if ret:
            return true_positives, false_positives, cumulative_true_positives, cumulative_false_positives

    def compute_precision_recall(self, verbose=True, ret=False):
        '''
        计算所有类的精度和召回率。注意：运行该方法前必须先运行match_predictions()。
        Arguments:
            verbose (bool, optional): If `True`, 运行时打印进度。
            ret (bool, optional): If `True`, 返回精度和召回率。
        Returns:

        '''

        if (self.cumulative_true_positives is None) or (self.cumulative_false_positives is None):
            raise ValueError("True and false positives not available. You must run `match_predictions()` before you call this method.")

        if (self.num_gt_per_class is None):
            raise ValueError("Number of ground truth boxes per class not available. You must run `get_num_gt_per_class()` before you call this method.")

        cumulative_precisions = [[]]
        cumulative_recalls = [[]]

        # 迭代所有类。
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("Computing precisions and recalls, class {}/{}".format(class_id, self.n_classes))

            tp = self.cumulative_true_positives[class_id]    #真正
            fp = self.cumulative_false_positives[class_id]   #假正


            cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0) # 1D array with shape `(num_predictions,)`
            cumulative_recall = tp / self.num_gt_per_class[class_id] # 1D array with shape `(num_predictions,)`

            cumulative_precisions.append(cumulative_precision)
            cumulative_recalls.append(cumulative_recall)

        self.cumulative_precisions = cumulative_precisions
        self.cumulative_recalls = cumulative_recalls

        if ret:
            return cumulative_precisions, cumulative_recalls

    def compute_average_precisions(self, mode='sample', num_recall_points=11, verbose=True, ret=False):
        '''
        计算每一类的AP。注意：调用该方法前，compute_precision_recall()必须被调用。
        Arguments:
            mode (str, optional): 执行模式。
            num_recall_points (int, optional): 仅仅跟sample模式有关。
            verbose (bool, optional): If `True`, 打印运行进度。
            ret (bool, optional): If `True`, 返回AP。
        Returns:
        '''

        if (self.cumulative_precisions is None) or (self.cumulative_recalls is None):
            raise ValueError("Precisions and recalls not available. You must run `compute_precision_recall()` before you call this method.")

        if not (mode in {'sample', 'integrate'}):
            raise ValueError("`mode` can be either 'sample' or 'integrate', but received '{}'".format(mode))

        average_precisions = [0.0]

        # 迭代所有类。
        for class_id in range(1, self.n_classes + 1):

            if verbose:
                print("Computing average precision, class {}/{}".format(class_id, self.n_classes))

            cumulative_precision = self.cumulative_precisions[class_id]
            cumulative_recall = self.cumulative_recalls[class_id]
            average_precision = 0.0

            if mode == 'sample':

                for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):

                    cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]

                    if cum_prec_recall_greater_t.size == 0:
                        precision = 0.0
                    else:
                        precision = np.amax(cum_prec_recall_greater_t)

                    average_precision += precision

                average_precision /= num_recall_points

            elif mode == 'integrate':

                unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(cumulative_recall, return_index=True, return_counts=True)

                maximal_precisions = np.zeros_like(unique_recalls)
                recall_deltas = np.zeros_like(unique_recalls)

                for i in range(len(unique_recalls)-2, -1, -1):
                    begin = unique_recall_indices[i]
                    end = unique_recall_indices[i + 1]

                    maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]), maximal_precisions[i + 1])

                    recall_deltas[i] = unique_recalls[i + 1] - unique_recalls[i]

                average_precision = np.sum(maximal_precisions * recall_deltas)

            average_precisions.append(average_precision)

        self.average_precisions = average_precisions

        if ret:
            return average_precisions

    def compute_mean_average_precision(self, ret=True):
        '''
        计算每一类的MAP。注意：调用该方法前，compute_precision_recall()必须被调用。

        Arguments:
            ret (bool, optional):
        Returns:

        '''

        if self.average_precisions is None:
            raise ValueError("Average precisions not available. You must run `compute_average_precisions()` before you call this method.")

        mean_average_precision = np.average(self.average_precisions[1:]) # 第一个类时背景，跳过。
        self.mean_average_precision = mean_average_precision

        if ret:
            return mean_average_precision