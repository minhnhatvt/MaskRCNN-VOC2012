#Mask R-CNN for training and evaluating on PASCAL SBD 2012 dataset

This is a study project of the Computer Vision course at University of Science HCM-VNU.
This code is heavily based on https://github.com/matterport/Mask_RCNN


# Getting Started
If you want to validate our result or training by yourself. We've already setup all the process in the google colab link below 





# Training on PASCAL SBD


```

# Train a new model starting from pretrained ImageNet 
python main.py train --dataset=data/sbd --model=imagenet

# Continue training a model that you had trained earlier
python main.py train --dataset=data/sbd --model="path to weights.h5"

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python main.py train --dataset=/path/to/coco/ --model=last
```

You can also run the validation on the sbd_val subset by command:
```
python main+.py evaluate --dataset=data/sbd --model="path to weights.h5"
```

The training schedule, learning rate, and other hyperparameters should be set in `mrcnn/config.py`.



## Citation
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```



## Requirements
Python 3, TensorFlow 1.15, Keras 2.0.8 and pycocotools and other common packages listed in `requirements.txt`.

### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)
* Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
  and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  subsets. More details in the original [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).

If you use Docker, the code has been verified to work on
[this Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/).


## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

