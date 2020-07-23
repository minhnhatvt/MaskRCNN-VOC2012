import sys
import time
import numpy as np
import imgaug

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

PASCAL_VOC_CLASSES = ("aeroplane", "bicycle", "bird", "boat", "bottle",
		      "bus", "car", "cat", "chair", "cow", "diningtable",
        	      "dog", "horse", "motorbike", "person", "pottedplant",
		      "sheep", "sofa", "train", "tvmonitor")


PASCAL_VOC_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16,
                  17: 17, 18: 18, 19: 19, 20: 20}



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Dataset Statistics')

    parser.add_argument('--dataset', required=True,
                        metavar="/path to dataset",
                        help='Directory of the dataset')
    parser.add_argument('--subset', required=True,
                        metavar="/Subset of the dataset (train or val)",
                        help='type of subset ( train or val )')
    args = parser.parse_args()

    data_dir = args.dataset
    subset = args.subset

    class_map =PASCAL_VOC_LABEL_MAP

    coco = COCO("{}/pascal_sbd_{}.json".format(data_dir, subset))

    
    #print dataset statistics
    catIds = coco.getCatIds() #get all category (class) id
    final_imgIds = [] 
    final_annIds = []
    for class_id in catIds:

        print("***** Category ID: {} - Category name: {} *****".format(class_id, PASCAL_VOC_CLASSES[class_map[class_id] -1]))
        annotationIds = coco.getAnnIds(catIds=class_id) #get all annotation id belong to the class_id
        print("Number of objects: {}".format(len(annotationIds)))

        imgIds = coco.getImgIds(catIds=class_id) #get all image id that contain obbject of class_id
        print("Number of images that contain at least 1 object of category[{}]: {}\n".format(PASCAL_VOC_CLASSES[class_map[class_id] -1],len(imgIds)))
        #notice that one image may have multiple class_id so in order to get total number, we must remove the dupplicated
        final_imgIds.extend(imgIds)
        final_annIds.extend(annotationIds) #extend the total list
    
    print("Total annotated objects: {}".format(len(final_annIds)))
    print("Total images: {}".format(len(list(set(final_imgIds)))))

    
