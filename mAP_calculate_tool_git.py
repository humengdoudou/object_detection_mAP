# -*- coding: utf-8 -*-

# This is the python code for mAP calculation in object detection task, it follows the standard pascal voc format,
# and read the gt files as .xml format, the prediction files as .txt format.
# It will output the AP of each class, and mAP for the all classes
#
# the gt .xml format follows the standard pascal voc format, and each .xml has <size>, <width>, <height>, <object>
# <name>, <bndbox>, <xmin>, <ymin>, <xmax>, <ymax> .etc attributes
# the prediction .txt format is the self-defined format, which outputs 6 elements for each detected roi,
# and are organized as class_id, conf_score, xmin, ymin, xmax, ymax
#
# IMPORTANT: the forward pass will output all det_rois with score > 0 to maintain high recall rate,
#            so the score threshold is meaningless in mAP calculation
#
# running the code is straightforward:
# 1 put the gt .xml files into xml_gt folder;
# 2 put the predicted .txt files into txt_predict folder;
# 3 put the *.names & *.txt in root directory. *.txt specify the image_list you want to compute mAP, test.txt for default
# 4 simply run: python mAP_calculate_tool_git.py
#
# you will get not only mAP results, but also auc result of each class by image results
#
# NOTE: the mAP result is slightly different from mAP pascal voc tool, I recommend you to use the mAP_calculate_tool.py, 
#       since the reference also does not confirm its accuracy with pascal voc metric. 
#       the advantage of this scirpt is the visualization shown of the mAP results.
#
# reference:
# 1 https://github.com/MathGaron/mean_average_precision
#
# Author: hzhumeng01 2018-02-11


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import os
import xml.etree.ElementTree as ET

DEBUG = False


ROOT_PATH = "/home/hzhumeng01/python_tools/mAP_calculation_self"
CLASS_NAME_LIST = "clothes.names"
GT_FOLDER = "xml_gt"
PREDICT_FOLDER = "txt_predict"
mAP_CALCULATE_IMAGE_LIST = "test.txt"
XML_EXTS = ".xml"
TXT_EXTS = ".txt"
DET_CLASSES = ["coat", "pants", "glasses", "hat", "shoes", "bag"]    # for mAP list plot output

"""
    Bounding box intersection over union calculation.
    Borrowed from pytorch SSD implementation : https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
    and adapted to numpy.
"""
def intersect_area(box_a, box_b):
    """
    Compute the area of intersection between two rectangular bounding box
    Bounding boxes use corner notation : [x1, y1, x2, y2]
    Args:
      box_a: (np.array) bounding boxes, Shape: [A,4].
      box_b: (np.array) bounding boxes, Shape: [B,4].
    Return:
      np.array intersection area, Shape: [A,B].
    """
    resized_A = box_a[:, np.newaxis, :]
    resized_B = box_b[np.newaxis, :, :]
    max_xy = np.minimum(resized_A[:, :, 2:], resized_B[:, :, 2:])
    min_xy = np.maximum(resized_A[:, :, :2], resized_B[:, :, :2])

    diff_xy = (max_xy - min_xy)
    inter = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (np.array) Predicted bounding boxes,    Shape: [n_pred, 4]
        box_b: (np.array) Ground Truth bounding boxes, Shape: [n_gt, 4]
    Return:
        jaccard overlap: (np.array) Shape: [n_pred, n_gt]
    """
    inter = intersect_area(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
    area_a = area_a[:, np.newaxis]
    area_b = area_b[np.newaxis, :]
    union = area_a + area_b - inter
    return inter / union


def show_frame(pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, background=np.zeros((500, 500, 3)), show_confidence=True):
    """
    Plot the boundingboxes
    :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 4]
    :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
    :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
    :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 4]
    :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
    :return:
    """
    n_pred = pred_bb.shape[0]
    n_gt = gt_bb.shape[0]
    n_class = np.max(np.append(pred_classes, gt_classes)) + 1
    h, w, c = background.shape

    ax = plt.subplot("111")
    ax.imshow(background)
    cmap = plt.cm.get_cmap('hsv')

    confidence_alpha = pred_conf.copy()
    if not show_confidence:
        confidence_alpha.fill(1)

    for i in range(n_pred):
        x1 = pred_bb[i, 0] * w
        y1 = pred_bb[i, 1] * h
        x2 = pred_bb[i, 2] * w
        y2 = pred_bb[i, 3] * h
        rect_w = x2 - x1
        rect_h = y2 - y1
        # print(x1, y1)
        ax.add_patch(patches.Rectangle((x1, y1), rect_w, rect_h,
                                       fill=False,
                                       edgecolor=cmap(float(pred_classes[i]) / n_class),
                                       linestyle='dashdot',
                                       alpha=confidence_alpha[i]))

    for i in range(n_gt):
        x1 = gt_bb[i, 0] * w
        y1 = gt_bb[i, 1] * h
        x2 = gt_bb[i, 2] * w
        y2 = gt_bb[i, 3] * h
        rect_w = x2 - x1
        rect_h = y2 - y1
        ax.add_patch(patches.Rectangle((x1, y1), rect_w, rect_h,
                                       fill=False,
                                       edgecolor=cmap(float(gt_classes[i]) / n_class)))

    legend_handles = []
    for i in range(n_class):
        legend_handles.append(patches.Patch(color=cmap(float(i) / n_class), label="class : {}".format(i)))
    ax.legend(handles=legend_handles)
    plt.show()


"""
    Simple accumulator class that keeps track of True positive, False positive and False negative
    to compute precision and recall of a certain class
"""
class APAccumulator:
    def __init__(self):
        self.TP, self.FP, self.FN = 0, 0, 0


    def inc_good_prediction(self, value=1):
        self.TP += value


    def inc_bad_prediction(self, value=1):
        self.FP += value


    def inc_not_predicted(self, value=1):
        self.FN += value


    @property
    def precision(self):
        total_predicted = self.TP + self.FP
        if total_predicted == 0:
            total_gt = self.TP + self.FN
            if total_gt == 0:
                return 1.
            else:
                return 0.
        return float(self.TP) / total_predicted


    @property
    def recall(self):
        total_gt = self.TP + self.FN
        if total_gt == 0:
            return 1.
        return float(self.TP) / total_gt


    def __str__(self):
        str = ""
        str += "True positives : {}\n".format(self.TP)
        str += "False positives : {}\n".format(self.FP)
        str += "False Negatives : {}\n".format(self.FN)
        str += "Precision : {}\n".format(self.precision)
        str += "Recall : {}\n".format(self.recall)
        return str


class DetectionMAP:
    def __init__(self, n_class, pr_samples=11, overlap_threshold=0.5):
        """
        Running computation of average precision of n_class in a bounding box + classification task
        :param n_class:             quantity of class
        :param pr_samples:          quantification of threshold for pr curve
        :param overlap_threshold:   minimum overlap threshold
        """
        self.n_class = n_class
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.reset_accumulators()


    def reset_accumulators(self):
        """
        Reset the accumulators state
        TODO this is hard to follow... should use a better data structure
        total_accumulators : list of list of accumulators at each pr_scale for each class
        :return:
        """
        self.total_accumulators = []
        for i in range(len(self.pr_scale)):
            class_accumulators = []
            for j in range(self.n_class):
                class_accumulators.append(APAccumulator())
            self.total_accumulators.append(class_accumulators)


    def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
        """
        Update the accumulator for the running mAP evaluation.
        For exemple, this can be called for each images
        :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 4]
        :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
        :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
        :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 4]
        :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
        :return:
        """

        if pred_bb.ndim == 1:
            pred_bb = np.repeat(pred_bb[:, np.newaxis], 4, axis=1)
        for accumulators, r in zip(self.total_accumulators, self.pr_scale):
            if DEBUG:
                print("Evaluate pr_scale {}".format(r))
            self.evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, r, self.overlap_threshold)


    @staticmethod
    def evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, confidence_threshold, overlap_threshold=0.5):
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        pred_size = pred_classes.shape[0]
        IoU = None
        if pred_size != 0:
            IoU = DetectionMAP.compute_IoU(pred_bb, gt_bb, pred_conf, confidence_threshold)
            # mask irrelevant overlaps
            IoU[IoU < overlap_threshold] = 0

        # Score Gt with no prediction
        for i, acc in enumerate(accumulators):
            qty = DetectionMAP.compute_false_negatives(pred_classes, gt_classes, IoU, i)
            acc.inc_not_predicted(qty)

        # If no prediction are made, no need to continue further
        if len(pred_bb) == 0:
            return

        # Final match : 1 prediction per GT
        for i, acc in enumerate(accumulators):
            qty = DetectionMAP.compute_true_positive(pred_classes, gt_classes, IoU, i)
            acc.inc_good_prediction(qty)
            qty = DetectionMAP.compute_false_positive(pred_classes, pred_conf, confidence_threshold, gt_classes, IoU, i)
            acc.inc_bad_prediction(qty)
            if DEBUG:
                print(accumulators[i])


    @staticmethod
    def compute_IoU(prediction, gt, confidence, confidence_threshold):
        IoU = jaccard(prediction, gt)
        IoU[confidence < confidence_threshold, :] = 0
        return IoU


    @staticmethod
    def compute_false_negatives(pred_cls, gt_cls, IoU, class_index):
        if len(pred_cls) == 0:
            return np.sum(gt_cls == class_index)
        IoU_mask = IoU != 0
        # check only the predictions from class index
        prediction_masks = pred_cls != class_index
        IoU_mask[prediction_masks, :] = False
        # keep only gt of class index
        mask = IoU_mask[:, gt_cls == class_index]
        # sum all gt with no prediction of its class
        return np.sum(np.logical_not(mask.any(axis=0)))


    @staticmethod
    def compute_true_positive(pred_cls, gt_cls, IoU, class_index):
        IoU_mask = IoU != 0
        # check only the predictions from class index
        prediction_masks = pred_cls != class_index
        IoU_mask[prediction_masks, :] = False
        # keep only gt of class index
        mask = IoU_mask[:, gt_cls == class_index]
        # sum all gt with prediction of this class
        return np.sum(mask.any(axis=0))


    @staticmethod
    def compute_false_positive(pred_cls, pred_conf, conf_threshold, gt_cls, IoU, class_index):
        # check if a prediction of other class on class_index gt
        IoU_mask = IoU != 0
        prediction_masks = pred_cls == class_index
        IoU_mask[prediction_masks, :] = False
        mask = IoU_mask[:, gt_cls == class_index]
        FP_predicted_by_other = np.sum(mask.any(axis=0))

        IoU_mask = IoU != 0
        prediction_masks = pred_cls != class_index
        IoU_mask[prediction_masks, :] = False
        gt_masks = gt_cls != class_index
        IoU_mask[:, gt_masks] = False
        # check if more than one prediction on class_index gt
        mask_double = IoU_mask[pred_cls == class_index, :]
        detection_per_gt = np.sum(mask_double, axis=0)
        FP_double = np.sum(detection_per_gt[detection_per_gt > 1] - 1)
        # check if class_index prediction outside of class_index gt
        # total prediction of class_index - prediction matched with class index gt
        detection_per_prediction = np.logical_and(pred_conf >= conf_threshold, pred_cls == class_index)
        FP_predict_other = np.sum(detection_per_prediction) - np.sum(detection_per_gt)
        return FP_double + FP_predict_other + FP_predicted_by_other


    @staticmethod
    def multiple_prediction_on_gt(IoU_mask, gt_classes, accumulators):
        """
        Gt with more than one overlap get False detections
        :param prediction_confidences:
        :param IoU_mask: Mask of valid intersection over union  (np.array)      IoU Shape [n_pred, n_gt]
        :param gt_classes:
        :param accumulators:
        :return: updated version of the IoU mask
        """
        # compute how many prediction per gt
        pred_max = np.sum(IoU_mask, axis=0)
        for i, gt_sum in enumerate(pred_max):
            gt_cls = gt_classes[i]
            if gt_sum > 1:
                for j in range(gt_sum - 1):
                    accumulators[gt_cls].inc_bad_prediction()


    def compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision


    def compute_precision_recall_(self, class_index, interpolated=True):
        precisions = []
        recalls = []
        for acc in self.total_accumulators:
            precisions.append(acc[class_index].precision)
            recalls.append(acc[class_index].recall)

        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions, recalls


    def plot_pr(self, ax, class_index, precisions, recalls, average_precision):
        ax.step(recalls, precisions, color='b', alpha=0.2,
                where='post')
        ax.fill_between(recalls, precisions, step='post', alpha=0.2,
                        color='b')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('cls {0:} : AUC={1:0.2f}'.format(class_index, average_precision))
        #ax.set_title('cls {0:} : AUC={1:0.2f}'.format(DET_CLASSES[class_index], average_precision))


    def plot(self, interpolated=True):
        """
        Plot all pr-curves for each classes
        :param interpolated: will compute the interpolated curve
        :return:
        """
        grid = int(math.ceil(math.sqrt(self.n_class)))
        fig, axes = plt.subplots(nrows=grid, ncols=grid)
        mean_average_precision = []
        # TODO: data structure not optimal for this operation...
        for i, ax in enumerate(axes.flat):
            if i > self.n_class - 1:
                break
            precisions, recalls = self.compute_precision_recall_(i, interpolated)
            average_precision = self.compute_ap(precisions, recalls)
            self.plot_pr(ax, i, precisions, recalls, average_precision)
            mean_average_precision.append(average_precision)
            print "mAP {}: {}".format(DET_CLASSES[i], average_precision)

        print "mAP:" + str(sum(mean_average_precision) / len(mean_average_precision))
        plt.suptitle("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))
        fig.tight_layout()


def load_class_names(filename, dirname):
    """
    load class names from text file
    :param filename: str, class filename
    :param dirname: str, file directory
    :return: dict of detection class
    """
    full_path = os.path.join(dirname, filename)
    assert os.path.exists(full_path), 'Path does not exist: {}'.format(full_path)
    with open(full_path, 'r') as f:
        classes_list = [l.strip() for l in f.readlines()]

    # list -> dict
    class_dict = {}
    for i, class_name in enumerate(classes_list):
        class_dict[class_name] = i

    return class_dict


def parse_gt_xml(filename, classes_dict):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :param classes_dict: detection class dict
    :return: array of class_id, gt_bbox, image_wid, image_hgt
    comment: an image may contains one more objects, multi objects construct array
    """
    tree = ET.parse(filename)
    size = tree.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)

    class_id_list = []
    bbox_total_list = []

    for obj in tree.findall('object'):
        class_name = obj.find('name').text    # class name, like coat, pants
        class_id = classes_dict[class_name]

        bbox = obj.find('bndbox')
        bbox_list = [float(bbox.find('xmin').text) / width,
                     float(bbox.find('ymin').text) / height,
                     float(bbox.find('xmax').text) / width,
                     float(bbox.find('ymax').text) / height]

        class_id_list.append(class_id)
        bbox_total_list.append(bbox_list)

    class_id_array = np.array(class_id_list)
    bbox_total_array = np.array(bbox_total_list)

    return class_id_array, bbox_total_array, width, height


def parse_predict_txt(file_name, width, height):
    """
    parse predict .txt file into some arrays
    :param file_name: .txt file path
    :param width: image width
    :param height: image height
    :return: array of pred_id, pred_score, pred_bbox
    comment: an image may contains one more objects, multi objects construct arrays
    """
    with open(file_name) as f:
        predict_rois = [x.strip() for x in f.readlines()]

    pred_id_list = []
    pred_score_list = []
    pred_bbox_list = []

    for i, pred_roi in enumerate(predict_rois):
        pred_list = pred_roi.strip().split(" ")

        pred_id_list.append(int(pred_list[0]))
        pred_score_list.append(float(pred_list[1]))

        xmin = float(pred_list[2]) / width
        ymin = float(pred_list[3]) / height
        xmax = float(pred_list[4]) / width
        ymax = float(pred_list[5]) / height

        pred_bbox_list.append([xmin, ymin, xmax, ymax])

    pred_id_array = np.array(pred_id_list)
    pred_score_array = np.array(pred_score_list)
    pred_bbox_array = np.array(pred_bbox_list)

    return pred_id_array, pred_score_array, pred_bbox_array


def get_frame_list(root_path, gt_folder, predict_folder, eval_image_file, classes_dict):
    """
    get mAP evaluate format, each gt\pred constructs a tuple, all tuples construct a list
    :param root_path: root path of the project
    :param gt_folder: gt .xml folder
    :param predict_folder: predict .txt folder
    :param eval_image_file: evaluate image name list file
    :param classes_dict: detection class dict
    :return: list of (pred_bbox, pred_class_id, pred_score, gt_bbox, gt_class_id)
    comment: an image may contains one more objects, multi objects construct arrays
    """
    # eval image list
    eval_img_path = os.path.join(root_path, eval_image_file)
    assert os.path.exists(eval_img_path), 'Path does not exist: {}'.format(eval_img_path)
    with open(eval_img_path) as f:
        eval_image_list = [x.strip() for x in f.readlines()]

    calculate_frames_list = []
    for i, eval_img_name in enumerate(eval_image_list):
        gt_xml_file_path = os.path.join(root_path, gt_folder, (eval_img_name + XML_EXTS))
        assert os.path.exists(gt_xml_file_path), 'Path does not exist: {}'.format(gt_xml_file_path)
        gt_class_id_array, gt_bbox_array, width, height = parse_gt_xml(gt_xml_file_path, classes_dict)

        pred_txt_file_path = os.path.join(root_path, predict_folder, (eval_img_name + TXT_EXTS))
        assert os.path.exists(pred_txt_file_path), 'Path does not exist: {}'.format(pred_txt_file_path)
        pred_class_id_array, pred_score_array, pred_bbbox_array = parse_predict_txt(pred_txt_file_path, width, height)

        img_tuple = (pred_bbbox_array, pred_class_id_array, pred_score_array, gt_bbox_array, gt_class_id_array)
        calculate_frames_list.append(img_tuple)

    return calculate_frames_list


if __name__ == '__main__':

    classes_dict = load_class_names(CLASS_NAME_LIST, ROOT_PATH)
    n_classes = len(classes_dict)

    mAP = DetectionMAP(n_classes)  # initialization

    calculate_frames_list = get_frame_list(ROOT_PATH, GT_FOLDER, PREDICT_FOLDER, mAP_CALCULATE_IMAGE_LIST, classes_dict)

    for i, frame in enumerate(calculate_frames_list):
        print("Evaluate frame {}".format(i))
        #show_frame(*frame)
        mAP.evaluate(*frame)

    mAP.plot()
    plt.show()
    #plt.savefig("pr_curve_example.png")