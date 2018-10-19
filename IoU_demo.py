# -*- coding: utf-8 -*-
#
# This is the python code for calculating bbox IoU,
# By running the script, we can get the IoU score between pred / gt bboxes
#
# Author: hzhumeng01 2018-10-19
# copyright @ netease, AI group


from __future__ import print_function, absolute_import
import numpy as np


def get_IoU(pred_bbox, gt_bbox):
    """
    return iou score between pred / gt bboxes
    :param pred_bbox: predict bbox coordinate
    :param gt_bbox: ground truth bbox coordinate
    :return: iou score
    """

    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(pred_bbox[2] - pred_bbox[0]) > 0) and
    #         (abs(pred_bbox[3] - pred_bbox[1]) > 0))
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and
    #         (abs(gt_bbox[3] - gt_bbox[1]) > 0))

    # -----0---- get coordinates of inters
    ixmin = max(pred_bbox[0], gt_bbox[0])
    iymin = max(pred_bbox[1], gt_bbox[1])
    ixmax = min(pred_bbox[2], gt_bbox[2])
    iymax = min(pred_bbox[3], gt_bbox[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # -----1----- intersection
    inters = iw * ih

    # -----2----- union, uni = S1 + S2 - inters
    uni = ((pred_bbox[2] - pred_bbox[0] + 1.) * (pred_bbox[3] - pred_bbox[1] + 1.) +
           (gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) -
           inters)

    # -----3----- iou
    overlaps = inters / uni

    return overlaps


def get_max_IoU(pred_bboxes, gt_bbox):
    """
    given 1 gt bbox, >1 pred bboxes, return max iou score for the given gt bbox and pred_bboxes
    :param pred_bbox: predict bboxes coordinates, we need to find the max iou score with gt bbox for these pred bboxes
    :param gt_bbox: ground truth bbox coordinate
    :return: max iou score
    """

    # bbox should be valid, actually we should add more judgements, just ignore here...
    # assert ((abs(gt_bbox[2] - gt_bbox[0]) > 0) and
    #         (abs(gt_bbox[3] - gt_bbox[1]) > 0))

    if pred_bboxes.shape[0] > 0:
        # -----0---- get coordinates of inters, but with multiple predict bboxes
        ixmin = np.maximum(pred_bboxes[:, 0], gt_bbox[0])
        iymin = np.maximum(pred_bboxes[:, 1], gt_bbox[1])
        ixmax = np.minimum(pred_bboxes[:, 2], gt_bbox[2])
        iymax = np.minimum(pred_bboxes[:, 3], gt_bbox[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        # -----1----- intersection
        inters = iw * ih

        # -----2----- union, uni = S1 + S2 - inters
        uni = ((gt_bbox[2] - gt_bbox[0] + 1.) * (gt_bbox[3] - gt_bbox[1] + 1.) +
               (pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1.) * (pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1.) -
               inters)

        # -----3----- iou, get max score and max iou index
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    return overlaps, ovmax, jmax


if __name__ == "__main__":

    # test1
    pred_bbox = np.array([50, 50, 90, 100])   # top-left: <50, 50>, bottom-down: <90, 100>, <x-axis, y-axis>
    gt_bbox = np.array([70, 80, 120, 150])

    print (get_IoU(pred_bbox, gt_bbox))

    # test2
    pred_bboxes = np.array([[15, 18, 47, 60],
                          [50, 50, 90, 100],
                          [70, 80, 120, 145],
                          [130, 160, 250, 280],
                          [25.6, 66.1, 113.3, 147.8]])
    gt_bbox = np.array([70, 80, 120, 150])
    print (get_max_IoU(pred_bboxes, gt_bbox))
