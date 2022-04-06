# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import visual_genome.local as vg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image

from AttributeROIHead import *

data_path = '/ssd/tobias/datasets/vg/'

## Run VG_Prepocess to obtain these files.
def get_vg_dicts_cached(vg_dir, val=False):
    fname = vg_dir + "detectron_" + ("val" if val else "train") + "_filtered.json"
    fhandle = open(fname)
    data = json.load(fhandle)
    fhandle.close()
    return data

appearing_objects = json.load(open("data/json/appearing_objects_filtered.json"))

# Register dataset.
for d in ["train", "val"]:
    DatasetCatalog.register("vg_" + d, lambda d=d: get_vg_dicts_cached(data_path, d == "val"))
    MetadataCatalog.get("vg_" + d).set(thing_classes=list(appearing_objects.values()))

vg_metadata = MetadataCatalog.get("vg_val")

# Set config and launch training.

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = "output/frcnn"
cfg.DATASETS.TRAIN = ("vg_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # "./output_new/model_final.pth" 
# Let training initialize from model zoo or start with last trained checkpoint
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
cfg.SOLVER.MAX_ITER = 100000  # 80000 to 100k iterations
cfg.SOLVER.STEPS = [80000, 90000]  # do not decay learning rate
# Use custom ROI_HEAD
cfg.MODEL.ROI_HEADS.NAME = "AttributeROIHead"
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3434  # Number of classes in trainingset
cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES = 2979  # number of attributes in trainingset
cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 3072 # Change the prediction layers to have dimension 3072
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = TrainerWithAttributes(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

