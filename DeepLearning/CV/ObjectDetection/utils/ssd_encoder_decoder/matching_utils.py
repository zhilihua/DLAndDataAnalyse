from __future__ import division
import numpy as np

def match_bipartite_greedy(weight_matrix):
    """
    :param weight_matrix: 是真实包围框和每一个锚框的iou矩阵
    :return:
    """
    weight_matrix = np.copy(weight_matrix)   #加入真实框5，锚框6个:[[1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]]
    num_ground_truth_boxes = weight_matrix.shape[0]   #真实包围框的数量，假如5个
    all_gt_indices = list(range(num_ground_truth_boxes))   #在上面假设情况下：[0, 1, 2, 3, 4]

    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)   #在上面成立情况下：[0, 0, 0, 0, 0]

    for _ in range(num_ground_truth_boxes):
        anchor_indices = np.argmax(weight_matrix, axis=1)   #选取每一个真实框最匹配的锚点索引共5个[3, 1, 1, 2, 4]。
        overlaps = weight_matrix[all_gt_indices, anchor_indices]   #选取每一个真实框最匹配的锚点的iou共5个[0.5, 0.1, 0.1, 0.2, 0.4]
        ground_truth_index = np.argmax(overlaps)   #选出iou最高的索引0.
        anchor_index = anchor_indices[ground_truth_index]  #选出匹配最高的锚点的索引3.
        matches[ground_truth_index] = anchor_index    #将真实包围框对应的索引的位置写上其最对应的锚框的索引

        weight_matrix[ground_truth_index] = 0   #将匹配最高的真实框与所有锚点的iou置0， 即删除该真实框
        weight_matrix[:, anchor_index] = 0      #将与该最高真实框匹配的锚点框对应的其他真实框对应概率置0，即删除该锚点框

    return matches  #每一个元素索引代表行，值代表列

def match_multi(weight_matrix, threshold):
    """
    :param weight_matrix: 是真实包围框和每一个锚框的iou矩阵[[1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]]
    :param threshold: 一个阈值假如0.5
    :return:
    """
    num_anchor_boxes = weight_matrix.shape[1]   #锚点框的个数6
    all_anchor_indices = list(range(num_anchor_boxes))  #进行一个排序[0, 1, 2, 3, 4，5]

    ground_truth_indices = np.argmax(weight_matrix, axis=0)   #选出每个锚点框最匹配的真实框[2, 4, 3, 3, 1, 1]
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]  #返回每一个锚点框对应的最匹配真实框的iou[0.6, 0.3, 0.8, 0.9, 0.4, 0.4]

    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]    #返回所有大于阈值的锚点框的索引，[0]是因为该函数格式为(array([0, 2], dtype=int64),)需要
                                                                        #第0个值，即array([0, 2], dtype=int64)，对应列
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]  #返回上述锚点框所对应真实框的索引，对应行

    return gt_indices_thresh_met, anchor_indices_thresh_met   #返回weight_matrix的行和列索引值