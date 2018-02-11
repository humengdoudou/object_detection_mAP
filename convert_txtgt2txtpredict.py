# -*- coding: utf-8 -*-

# This is the python code for converting txt_gt .txt file format to txt_predict .txt file format
#
# Author: hzhumeng01 2018-02-07

from __future__ import print_function, absolute_import

import os

LABEL_INDEX_DICT = {'coat':0,
                    'pants':1,
                    'glasses':2,
                    'hat':3,
                    'shoes':4,
                    'bag':5}

TXT_EXTS = ".txt"
ROOT_PATH = "/home/hzhumeng01/python_tools/mAP_calculation_self"

def load_image_set(root_path, image_list_name):
    """
    find out which indexes correspond to given image set

    Parameters:
    ----------

    Returns:
    ----------
    entire list of images specified in the setting
    """
    image_list_path = os.path.join(root_path, image_list_name)
    assert os.path.exists(image_list_path), 'Path does not exist: {}'.format(image_list_path)
    with open(image_list_path) as f:
        image_name_list = [x.strip() for x in f.readlines()]

    return image_name_list


if __name__ == "__main__":

    image_file_name = "test.txt"
    txt_gt_folder = "txt_gt"
    txt_predict_folder = "txt_predict"

    if not os.path.isdir(os.path.join(ROOT_PATH, txt_predict_folder)):
        os.mkdir(os.path.join(ROOT_PATH, txt_predict_folder))

    image_name_list = load_image_set(ROOT_PATH, image_file_name)

    for ind, image_filename in enumerate(image_name_list):
        txt_gt_file = os.path.join(ROOT_PATH, txt_gt_folder, (image_filename + TXT_EXTS))
        assert os.path.exists(txt_gt_file), 'Path does not exist: {}'.format(txt_gt_file)
        with open(txt_gt_file) as f_read:
            gt_results = [x.strip() for x in f_read.readlines()]    # read gt_result in each .txt of txt_gt

        count = 0
        predict_str_list = []
        for each_line in gt_results:
            if count == 0:  # first line, just the object num
                count += 1
                continue

            gt_result = each_line.strip().split(" ")
            gt_result[0] = LABEL_INDEX_DICT[gt_result[0]]   # replace class str to class_id index, like "coat" -> 0

            predict_str = "{} {} {} {} {} {}\n".format(gt_result[0],
                                                       1.0,
                                                       gt_result[1],
                                                       gt_result[2],
                                                       gt_result[3],
                                                       gt_result[4])

            predict_str_list.append(predict_str)

        with open(os.path.join(ROOT_PATH, txt_predict_folder, (image_filename + TXT_EXTS)), 'wt') as f_write:
            f_write.writelines(predict_str_list)
        f_write.close()
