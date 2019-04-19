import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco.coco as coco
import utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize

import cv2
#%matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH =  "mask_rcnn_coco.h5"


# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
file_names = IMAGE_DIR +"/7933423348_c30bd9bd4e_z.jpg"
image = skimage.io.imread(file_names)
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
# print(r['rois'][0])
# print(r['masks'][1][2])
# print(r['masks'].shape[2])       #r['masks'].shape[2]用來判斷有幾個物體 r['masks'].shape[2]=2代表有兩個物體被偵測 

img = np.zeros((image.shape[0],image.shape[1],image.shape[2]),dtype = int)
#抓最顯著的人的mask 花12.4秒
for a in range(r['class_ids'].shape[0]):
  if r['class_ids'][a]==1:
    for i in range(r['rois'][a][0],r['rois'][a][2]):
        for j in range(r['rois'][a][1],r['rois'][a][3]):
          if r['masks'][i][j][a] == True:
            img[i][j] = 255
    break

#抓所有人的mask 花12.6秒
# for a in range(r['class_ids'].shape[0]):
#   if r['class_ids'][a]==1:
#     for i in range(r['rois'][a][0],r['rois'][a][2]):
#         for j in range(r['rois'][a][1],r['rois'][a][3]):
#           if r['masks'][i][j][a] == True:
#             img[i][j] = 255


#抓所有物體的mask 花17.8秒
# for k in range(r['masks'].shape[2]):   
#   for i in range(r['masks'].shape[0]):
#     for j in range(r['masks'].shape[1]):
#         if(r['masks'][i][j][k]==True):
#           img[i][j] = 255

cv2.imwrite("a.jpg",img)

# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                             class_names, r['scores'])
# plt.show()