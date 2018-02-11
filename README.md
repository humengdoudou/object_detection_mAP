# object detection mAP metric


tag： object detection mAP metric

---

### 1. Introduction


This is the python code for mAP calculation in object detection task, it follows the standard pascal voc format,
and read the gt files as .xml format, the prediction files as .txt format.

It will output the AP of each class, and mAP for the all classes

the gt .xml format follows the standard pascal voc format, and each .xml has 
```
<size>, <width>, <height>, <object>, <name>, <bndbox>, <xmin>, <ymin>, <xmax>, <ymax> .etc 
```
attributes.

The prediction .txt format is the self-defined format, which outputs 6 elements for each detected roi,
and are organized as 
```
class_id, conf_score, xmin, ymin, xmax, ymax
```

**IMPORTANT**: 
```
In general training procedure, the mAP calculation will use all det_rois with conf_score > 0 to maintain high recall rate.

So the testing procedure follows the training schedule, and save all det_rois without score_thres and overlap_thres restrict.
```

### 2. How to do

running the code is straightforward:

1. put the gt .xml files into xml_gt folder;
2. put the predicted .txt files into txt_predict folder;
3. put the *.names & *.txt in root directory. *.txt specify the image_list you want to compute mAP, test.txt for default
4. simply run: python mAP_calculate_tool.py

**IMPORTANT**: 

the .pkl files in cache_gt_pkl and predict.pkl, are used for increasing calculation speed. So if you change the gt .xml files or predict .txt files,
you need to update the .pkl files. 

The simplest way to handle this is **delete** all this .pkl files.

### 3. references

1. https://github.com/zhreshold/mxnet-ssd/tree/master/evaluate
2. https://github.com/zhreshold/mxnet-ssd/blob/master/dataset/pascal_voc.py
3. http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html
4. https://sanchom.wordpress.com/2011/09/01/precision-recall/

### 4. 修订明细

| revise indx  |  revise time |  revise version  |  reviser  | revise comments |
| :-----:      | :-----:      | :----:           | :-----:   | :----:          |
| 1            | 2018-02-06   |   V1.0           |   humengdoudou    |          |
| 2            |              |                  |           |                 |
| 3            |              |                  |           |                 |
| 4            |              |                  |           |                 |
