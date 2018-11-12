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
# 4 simply run: python mAP_calculate_tool_pascalvoc.py
#
# NOTE: the mAP result follows the official pascal voc standard, and its result is reliable
#
# Author: hzhumeng01 2018-02-06
# copyright @ netease, AI group


from __future__ import print_function, absolute_import

import os
import numpy as np
import xml.etree.ElementTree as ET

try:
    import cPickle as pickle
except ImportError:
    import pickle


IOU_THRES = 0.5


def parse_predict_txt(txt_folder, txt_file_name, exts = ".txt"):
    """
    parse predict .txt file into a list
    :param txt_folder: .txt folder path
    :param txt_file_name: .txt file name, no .exts
    :param exts: predict file exts, ".txt" for default
    :return: list of predict_rois,
             example: ["class_id1 score1 xmin1 ymin1 xmax1 ymax1", "class_id2 score2 xmin2 ymin2 xmax2 ymax2"]
    comment: an image may contains one more objects, one object constructs a predict_roi string,
             and objects construct predict_rois string list
    """
    predict_txt = os.path.join(txt_folder, (txt_file_name + exts))
    assert os.path.exists(predict_txt), 'Path does not exist: {}'.format(predict_txt)
    with open(predict_txt) as f:
        predict_rois = [x.strip() for x in f.readlines()]

    return predict_rois


def parse_gt_xml(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    comment: an image may contains one more objects, one object constructs an obj_dict,
             and obj_dicts construct objects_list
    """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()

        if obj.find('name').text is None:
            raise Exception("None class label")

        obj_dict['name'] = obj.find('name').text    # class name, like coat, pants
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_dict)
    return objects


def ap_calculate(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changessihu
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_eachclass(filename, xml_gt_file, mAP_calcu_image_list, classname, cache_dir, ovthresh=0.5, use_07_metric=False):
    """
    compute average precision for each classname
    :param filename: detection results saved in txt_predict_class, like predict_bag.txt, filename.format(classname)
    :param xml_gt_file: gt .xml files string. xml_gt_file.format(image_filename)
    :param mAP_calcu_image_list: text file containing list of images for computing AP
    :param classname: category name fot AP
    :param cache_dir: caching annotations for each classname
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, (classname + '.pkl'))
    with open(mAP_calcu_image_list, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # do not need to store the cache files, 20180412, comment by hzhumeng01
    # load annotations from cache, recs contains all the gt .xml files
    # if not os.path.isfile(cache_file):
    #     gt_dict = {}
    #     for ind, image_filename in enumerate(image_filenames):
    #         gt_dict[image_filename] = parse_gt_xml(xml_gt_file.format(image_filename))
    #         #if ind % 100 == 0:
    #         #    print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames)))
    #     #print('saving annotations cache to {:s}'.format(cache_file))
    #     with open(cache_file, 'wb') as f:
    #         pickle.dump(gt_dict, f)
    # else:
    #     with open(cache_file, 'rb') as f:
    #         gt_dict = pickle.load(f)

    # do not need to store the cache files, comment above, and just read, 20180412, comment by hzhumeng01
    gt_dict = {}
    for ind, image_filename in enumerate(image_filenames):
        gt_dict[image_filename] = parse_gt_xml(xml_gt_file.format(image_filename))

    # -----gt-----extract objects in :param classname:
    bbox_dict = {}
    npos = 0
    for image_filename in image_filenames:
        objects = [obj for obj in gt_dict[image_filename] if obj['name'] == classname]   # an image may has many same class rois
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        bbox_dict[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}

    # -----predictions-----read detections in classname files
    det_file = filename.format(classname)
    with open(det_file, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        return 0, 0, 0

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_inds = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    bbox = bbox[sorted_inds, :]
    image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        r = bbox_dict[image_ids[d]]
        bb = bbox[d, :].astype(float)
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not r['difficult'][jmax]:
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)

    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = ap_calculate(rec, prec, use_07_metric)

    return rec, prec, ap


class PascalVoc(object):
    """
    Implementation of Pascal VOC datasets

    Parameters:
    ----------
    root_path : str
        project root path
    gt_folder : str
        xml_gt folder name
    predict_folder : str
        txt_predict folder
    mAP_calcu_image_list : str
        .txt file, image list name for mAP
    class_name_list : str
        .names file for specifying the class names
    txt_predict_class_folder : str
        each class dets save folder
    use_07_metric : boolean
        use the pascal voc 07 metric or not
    """
    def __init__(self, root_path, gt_folder, predict_folder, mAP_calcu_image_list, class_name_list,
                 txt_predict_class_folder = "txt_predict_class", use_07_metric = False):
        super(PascalVoc, self).__init__()
        self.root_path = root_path
        self.gt_folder = gt_folder
        self.predict_folder = predict_folder
        self.mAP_calcu_image_list = mAP_calcu_image_list
        self.class_name_list = class_name_list
        self.txt_predict_class_folder = txt_predict_class_folder
        self.use_07_metric = use_07_metric

        self.classes_list = self._load_class_names(self.class_name_list, self.root_path)

        self.num_classes = len(self.classes_list)
        self.image_name_list = self.load_image_set()
        self.num_images = len(self.image_name_list)


    @property
    def cache_path(self):
        """
        make a directory to store all caches
        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), 'cache_gt_pkl')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path


    def _load_class_names(self, filename, dirname):
        """
        load class names from text file

        Parameters:
        ----------
        filename: str
            file stores class names
        dirname: str
            file directory
        """
        full_path = os.path.join(dirname, filename)
        classes_list = []
        with open(full_path, 'r') as f:
            classes_list = [l.strip() for l in f.readlines()]

        return classes_list


    def load_image_set(self):
        """
        find out which indexes correspond to given image set (train or val or test)

        Parameters:
        ----------

        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.root_path, self.mAP_calcu_image_list)
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_name_list = [x.strip() for x in f.readlines()]

        return image_name_list


    def evaluate_detections(self, predict_dict):
        """
        top level evaluations
        Parameters:
        ----------
        predict_dict: dict
            predict results, format can refer to the __main__
        Returns:
        ----------
            None
        """
        # make all these folders for results, for computing AP and mAP

        result_dir = os.path.join(self.root_path, self.txt_predict_class_folder)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_pascal_results(predict_dict)
        self.do_python_eval()


    def get_result_file_template(self):
        """
        this is a template, sub directory filename looks like: predict_coat.txt

        Returns:
        ----------
            a string template
        """
        res_file_folder = os.path.join(self.root_path, self.txt_predict_class_folder)
        filename = 'predict' + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path


    def write_pascal_results(self, predict_dict):
        """
        write results files in pascal devkit path
        Parameters:
        ----------
        predict_dict: dict
            boxes to be processed,
            each dict looks like ["class_id1 score1 xmin1 ymin1 xmax1 ymax1", "class_id2 score2 xmin2 ymin2 xmax2 ymax2"]
        Returns:
        ----------
        None
        """

        for cls_ind, cls_name in enumerate(self.classes_list):
            # print('Writing {} VOC results file'.format(cls))
            filename = self.get_result_file_template().format(cls_name)
            with open(filename, 'wt') as f:
                for img_ind, img_name in enumerate(self.image_name_list):  # loop read whole test image_list
                    dets = predict_dict[img_name]
                    if len(dets) < 1:
                        continue
                    for k, det in enumerate(dets):
                        roi_list = det.strip().split(" ")

                        if (int(roi_list[0]) == cls_ind):
                        # if (roi_list[0] == self.classes_list[cls_ind]):
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(img_name,
                                           float(roi_list[1]),
                                           float(roi_list[2]),
                                           float(roi_list[3]),
                                           float(roi_list[4]),
                                           float(roi_list[5])))


    def do_python_eval(self):
        """
        python evaluation wrapper, for computing AP and mAP

        Returns:
        ----------
        None
        """
        xml_gt_file = os.path.join(self.root_path, self.gt_folder, '{:s}.xml')
        mAP_calcu_image_list = os.path.join(self.root_path, self.mAP_calcu_image_list)

        aps = []
        print('VOC07 metric? ' + ('Y' if self.use_07_metric else 'No'))
        for cls_ind, cls in enumerate(self.classes_list):
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = ap_eachclass(filename, xml_gt_file, mAP_calcu_image_list, cls, self.cache_path,
                                         ovthresh=IOU_THRES, use_07_metric=self.use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))


if __name__ == "__main__":

    # -----0----- params
    # root_path = "/home/hzhumeng01/logo_detect/object_detection_mAP_pascalvoc_self"
    # gt_folder = "gt/xml_gt_5000_V4"      #  "xml_gt_5000"
    # predict_folder = "20180419/txt_predict_5000_V4"    # "txt_predict_5000"
    # mAP_calcu_image_list = "test_list/test_5000_V4.txt"   # "test_5000.txt"
    # class_name_list = "logo.names"
    # txt_predict_class_folder = "txt_predict_class"
    # use_07_metric = False  #True  False


    root_path = "/home/hzhumeng01/logo_detect/object_detection_mAP_pascalvoc_self"
    gt_folder = "gt/self_test_xml_1712"   # self_test_xml_1712  QA_test_xml_1895
    predict_folder = "pred/20181102/self_test_1712_txt" # QA_test_1895_txt  self_test_1712_txt
    mAP_calcu_image_list = "test_list/self_test_1712.txt"  # QA_test_1895 self_test_1712
    class_name_list = "logo.names"
    txt_predict_class_folder = "txt_predict_class"
    use_07_metric = False  # True  False


    # -----1----- initialize PascalVoc
    pascal_voc_style = PascalVoc(root_path,
                                 gt_folder,
                                 predict_folder,
                                 mAP_calcu_image_list,
                                 class_name_list,
                                 txt_predict_class_folder,
                                 use_07_metric)

    # -----2----- for predict.pkl
    predict_pkl_file = "predict.pkl"
    predict_pkl_path = os.path.join(root_path, predict_pkl_file)

    # load predicts from txt_predict folder, predict_recs contains all predicted .txt files
    # old
    # predict_dict = {}
    # if not os.path.isfile(predict_pkl_file):
    #     image_name_list = pascal_voc_style.load_image_set()
    #
    #     for ind, image_filename in enumerate(image_name_list):
    #         predict_dict[image_filename] = parse_predict_txt(os.path.join(root_path, predict_folder), image_filename, ".txt")
    #         # if ind % 100 == 0:
    #             # print('reading annotations for {:d}/{:d}'.format(ind + 1, len(image_name_list)))
    #     # print('saving annotations cache to {:s}'.format(predict_pkl_path))
    #     with open(predict_pkl_path, 'wb') as f:
    #         pickle.dump(predict_dict, f)
    # else:
    #     with open(predict_pkl_path, 'rb') as f:
    #         predict_dict = pickle.load(f)
    # old

    # new
    predict_dict = {}
    image_name_list = pascal_voc_style.load_image_set()

    for ind, image_filename in enumerate(image_name_list):
        predict_dict[image_filename] = parse_predict_txt(os.path.join(root_path, predict_folder),
                                                         image_filename, ".txt")

    # do not need to store the cache files, 20180412, comment by hzhumeng01
    # with open(predict_pkl_path, 'wb') as f:
    #     pickle.dump(predict_dict, f)
    # new

    # -----3----- calculate mAP
    pascal_voc_style.evaluate_detections(predict_dict)
