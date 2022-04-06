from re import S
from turtle import back
import torch
import pickle
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import clip
from PIL import Image
import numpy
from collections import OrderedDict
import json
import cv2
from concept_reader import Concept
from tqdm.utils import _environ_cols_wrapper

## For the FRCNN-Model
import detectron2
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
from AttributeROIHead import * # Modified Faster-RCNN for attribute prediction

# -- Code modified from source: https://github.com/kazuto1011/grad-cam-pytorch
class _BaseWrapper(object):
    """
    Please modify forward() and backward() depending on your task.
    """
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def generate(self):
        raise NotImplementedError

    def forward(self, image):
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return list(zip(*self.probs.sort(0, True)))  # element: (probability, index)

class GradCam(_BaseWrapper):
    """
    Compute GradCam feature map
    """
    def __init__(self, model, candidate_layers=[]):
        super(GradCam, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers
        # for module in self.model.named_modules():
        #     print(module[0])

        def forward_hook(module, input, output):
            self.fmap_pool[id(module)] = output.detach()


        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            if len(self.candidate_layers) == 0 or module[0] in self.candidate_layers:
                self.handlers.append(module[1].register_forward_hook(forward_hook))
                self.handlers.append(module[1].register_backward_hook(backward_hook))

    def find(self, pool, target_layer):
        # --- Query the right layer and return it's value.
        for key, value in pool.items():
            for module in self.model.named_modules():
                # print(module[0], id(module[1]), key)
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError(f"Invalid Layer Name: {target_layer}")

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads ,2))) + 1e-5
        return grads /l2_norm

    def compute_grad_weights(self, grads):
        grads = self.normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)


    def generate(self, target_layer):
        fmaps = self.find(self.fmap_pool, target_layer)
        grads = self.find(self.grad_pool, target_layer)
        weights = self.compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.0)

        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam

def compute_gradCAM(probs, labels, gcam, testing_labels, target_layer='layer4'):
    """
    Input:
        probs: Logits from the network
        labels: Label of the input
        gcam: See Class GradCam
        testing_labels: True for return the gradcam for label class; false for prediction class
        target_layer: Target layer name
    Return:
         probs: Logits from the network
         gcam_out: gcam map
         one_hot: one hot vector for the class of gcam
    """
    one_hot = torch.zeros((probs.shape[0], probs.shape[1])).float()
    max_int = torch.max(torch.nn.Sigmoid()(probs), 1)[1]

    if testing_labels:
        for i in range(one_hot.shape[0]):
            one_hot[i][max_int[i]] = 1.0

    else:
        for i in range(one_hot.shape[0]):
            one_hot[i][torch.max(labels, 1)[1][i]] = 1.0

    probs.backward(gradient=one_hot.to(gcam.device), retain_graph=True)
    fmaps = gcam.find(gcam.fmap_pool, target_layer)
    grads = gcam.find(gcam.grad_pool, target_layer)

    weights = torch.nn.functional.adaptive_avg_pool2d(grads, 1)
    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam_out = torch.nn.functional.relu(gcam)
    return probs, gcam_out, one_hot


def get_mask(gcam):
    """
    Input:
        gcam: gcam feature map in dimension of (B,N,H,W)
    Return:
        mask: gcam ready for visualization
    """
    for i in range(gcam.shape[0]):
        temp_loc = -1
        if gcam[i][:].sum() != 0:
            gcam[i][:] = gcam[i][:]
        else:
            temp_loc = i

        # if temp_loc != -1:
            # print('#--Zero SUM Error for image idx %d--#'%i)

    # gcam = torch.nn.functional.interpolate(gcam, size=(224,224), mode='bilinear', align_corners=False)
    B, C, H, W = gcam.shape
    gcam = gcam.view(B, -1)
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    gcam /= (gcam.max(dim=1, keepdim=True)[0])
    mask = gcam.view(B, C, H, W)

    return mask

### Helper function for FRCNN
def init_frcnn(data_path, model_file, device):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("vg_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 20000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate

    cfg.MODEL.ROI_HEADS.NAME = "AttributeROIHead"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3434  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.NUM_ATTRIBUTES = 2979  # number of attributes in trainingset
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 3072
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = "output"
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.WEIGHTS = os.path.join(model_file)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set a custom testing threshold!

    predictor = DefaultPredictor(cfg)

    appearing_objects = json.load(open(data_path + "appearing_objects_filtered.json"))
    appearing_attributes = json.load(open(data_path + "appearing_attributes_filtered.json"))
    
    for d in ["train", "val"]:
        #DatasetCatalog.register("vg_" + d, lambda d=d: get_vg_dicts_cached(data_path, d == "val"))
        MetadataCatalog.get("vg_" + d).set(thing_classes=list(appearing_objects.values()))
        MetadataCatalog.get("vg_" + d).set(attribute_classes=list(appearing_attributes.values()))
    vg_metadata = MetadataCatalog.get("vg_val")
    appearing_objects_dict = {int(k): v for k, v in appearing_objects.items()}
    return predictor, appearing_objects_dict


def compute_overlap_score(instances, saliency):
    """ Compute how good a saliency map and the detections overlap.
        Algorithm: Upsample saliency to resultion of the Bounding Boxes.
        Input: detected instances, return a {word, score} dictionary. If there are multiple detections, the score will be upgraded.
        instaces: Return from predict, which contains bounding boxes
        saliency: (1,N,N) tensor with the saliency map for the concept.
    """
    ret_dict = {}
    up_sampler = torch.nn.Upsample(size=(448,448), mode ="bilinear", align_corners=False)
    saliency = (saliency-torch.min(saliency))/(torch.max(saliency)-torch.min(saliency))
    up_sal = up_sampler(saliency.unsqueeze(0)).squeeze(0).squeeze(0)
    classes = instances.pred_classes
    attributes = instances.pred_attributes.tolist() 
    boxes = instances.pred_boxes.tensor
    ## boxes are in the x1, y1, x2, y2 format
    #print(classes)
    uclasses = torch.unique(classes)
    for idx in range(len(uclasses)): # unique classes
        class_mask = torch.zeros_like(up_sal)
        occurances = torch.where(classes == uclasses[idx])[0]
        #print(occurances)
        for occ_idx in range(len(occurances)):
            bbox = boxes[occurances[occ_idx]]
            class_mask[int(bbox[1].item()):int(bbox[2].item()), int(bbox[0].item()):int(bbox[2].item())] = 1.0
        # Calculate IoU score
        union = torch.max(torch.stack((up_sal, class_mask), dim=0), dim=0)[0]
        #print(union.shape)
        union = torch.sum(union)
        intersection = torch.min(torch.stack((up_sal, class_mask), dim=0), dim=0)[0]
        intersection = torch.sum(intersection)
        #print(intersection/union)
        ret_dict[uclasses[idx].item()] = intersection.item()/union.item()
    
    return ret_dict
    
def fuse_scores(dict_1, dict_2):
    """ Fuse dicts of detected objects and add up their scores """
    for k, v in dict_2.items():
        if k in dict_1:
            dict_1[k] += dict_2[k]
        else:
            dict_1[k] = dict_2[k]
    return dict_1
    

class Window_Crop(torch.nn.Module):
    """
    To crop the image based on (concept) activation map. We use sliding windows of different sizes, and find the one which 
    has the highest average
        N_pixel: the number of pixels used in  activation map for cropping
        device: GPU or CPU
        saliency_size: the size of activation map (W=H)
        input_size: the size of input image to be cropped (W=H)
    """
    def __init__(self, N_pixel, device, saliency_size, input_size):
        super(Window_Crop, self).__init__()

        self.N_pixel = N_pixel
        ##### some hypoparameters #######
        self.N_list = [1]
        self.proposalN = sum(self.N_list)  # proposal window num
        # iou_threshs = [0.25, 0.25, 0.25, 0.25]
        self.iou_threshs = [0.25]
        self.stride = input_size//saliency_size
        self.input_size = input_size
        # print(self.stride, self.input_size)
        self.device = device

        self.ratios, self.coordinates_cat, self.window_nums_sum, self.coordinates_feat_cat \
          = self.window_prepare(self.N_pixel, self.stride, self.input_size)

        self.avgpools = [torch.nn.AvgPool2d(self.ratios[i], 1) for i in range(len(self.ratios))]
        # self.avgpools = [torch.nn.AvgPool2d(ratios[i],stride) for i in range(len(ratios))]

    def get_information(self):
        return self.proposalN, self.coordinates_cat, self.coordinates_feat_cat

    def forward(self, x):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(self.ratios))]

        # feature map sum
        # fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]

        all_scores = torch.cat([avgs[i].view(batch, -1, 1) for i in range(len(self.ratios))], dim=1).to(self.device)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(self.device).reshape(batch, -1)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(self.window_nums_sum)-1):
                indices_results.append(self.nms(scores[sum(self.window_nums_sum[:j+1]):sum(self.window_nums_sum[:j+2])], proposalN=self.N_list[j], iou_threshs=self.iou_threshs[j],
                                           coordinates=self.coordinates_cat[sum(self.window_nums_sum[:j+1]):sum(self.window_nums_sum[:j+2])]) + sum(self.window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(numpy.concatenate(indices_results, 1))   # reverse

        proposalN_indices = numpy.array(proposalN_indices).reshape(batch, self.proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(self.device)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, self.proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores

    def compute_window_nums(self, ratios, stride, input_size):
        """ Return number of possible sliding windows for the given aspect ratio. """
        size = input_size / stride
        window_nums = []

        for _, ratio in enumerate(ratios):
            window_nums.append(int((size - ratio[0]) + 1) * int((size - ratio[1]) + 1))

        return window_nums

    def compute_coordinate(self, image_size, stride, indice, ratio):
        size = int(image_size / stride)
        column_window_num = (size - ratio[1]) + 1
        x_indice = indice // column_window_num
        y_indice = indice % column_window_num
        x_lefttop = x_indice * stride - 1
        y_lefttop = y_indice * stride - 1
        x_rightlow = x_lefttop + ratio[0] * stride
        y_rightlow = y_lefttop + ratio[1] * stride
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = numpy.array((x_lefttop, y_lefttop, x_rightlow, y_rightlow)).reshape(1, 4)

        return coordinate


    def indices2coordinates(self, indices, stride, image_size, ratio):
        batch, _ = indices.shape
        coordinates = []

        for j, indice in enumerate(indices):
            coordinates.append(self.compute_coordinate(image_size, stride, indice, ratio))

        coordinates = numpy.array(coordinates).reshape(batch,4).astype(int)       # [N, 4]
        return coordinates

    def indices2featmapcoords(self, indices, stride, image_size, ratio):
        batch, _ = indices.shape
        coordinates = []

        for j, indice in enumerate(indices):
            coordinates.append(self.compute_coordinate(image_size, stride, indice, ratio))

        coordinates = numpy.array(coordinates).reshape(batch,4).astype(int)       # [N, 4]
        return coordinates

    def window_prepare(self, N_pixel, stride, input_size):
        """ Compute sliding windows.
            N_pixel: Number of pixels that the crop should approx. have.
            stride: Stride, usually 1.
            Returns: i sizes of the crop (H; W), the coordinates each possible sliding window for each size, the  number of windows for each crop size,
        """
        ratio_base = int(torch.sqrt(N_pixel))
        ## Use five differente aspect ratios.
        ratios = [[ratio_base,ratio_base], 
                  [int(0.8*ratio_base),int(N_pixel//(0.8*ratio_base))], [int(N_pixel//(0.8*ratio_base)),int(0.8*ratio_base)], #[int(0.8*ratio_base),int(0.8*ratio_base)],
                  [int(1.2*ratio_base),int(N_pixel//(1.2*ratio_base))], [int(N_pixel//(1.2*ratio_base)),int(1.2*ratio_base)]] # [int(1.2*ratio_base),int(1.2*ratio_base)],

        # compute indice to coordinates
        window_nums = self.compute_window_nums(ratios, stride, input_size)
        indices_ndarrays = [numpy.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
        coordinates = [self.indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # convert indices to coordinates
        coordinates_feat = [self.indices2featmapcoords(indices_ndarray, 1, self.input_size//self.stride, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # convert indices coords to feature map
        coordinates_cat = numpy.concatenate(coordinates, 0)
        coordinates_feat_cat = numpy.concatenate(coordinates_feat, 0)
        # window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
        window_nums_sum = [0, sum(window_nums[:-1])]  # ,sum(window_nums[2:6]), sum(window_nums[6:])
        return ratios, coordinates_cat, window_nums_sum, coordinates_feat_cat

    def nms(self, scores_np, proposalN, iou_threshs, coordinates):
        """ 
        Non Maximum Suppression
        Input:
            scores_np: Sliding windows scores, list of scores []
            proposalN: The amount of sliding windows
            iou_threshs: Threshold for IOU
            coordinates: An array of coordinates (See window_prepare())
        Return:
            An array of index of the chosen window in coordinate array.
        """
        if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
            raise TypeError('score_np is not right')

        windows_num = scores_np.shape[0]
        indices_coordinates = numpy.concatenate((scores_np, coordinates), 1)
        #print(windows_num, indices_coordinates, scores_np.shape)
        indices = numpy.argsort(indices_coordinates[:, 0])
        indices_coordinates = numpy.concatenate((indices_coordinates, numpy.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
        indices_results = []

        res = indices_coordinates

        while res.any():
            indice_coordinates = res[-1]
            indices_results.append(indice_coordinates[5])

            if len(indices_results) == proposalN:
                return numpy.array(indices_results).reshape(1,proposalN).astype(numpy.int)
            res = res[:-1]

            # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
            start_max = numpy.maximum(res[:, 1:3], indice_coordinates[1:3])
            end_min = numpy.minimum(res[:, 3:5], indice_coordinates[3:5])
            lengths = end_min - start_max + 1
            intersec_map = lengths[:, 0] * lengths[:, 1]
            intersec_map[numpy.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
            iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                          (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                          (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
            res = res[iou_map_cur <= iou_threshs]

        while len(indices_results) != proposalN:
            indices_results.append(indice_coordinates[5])

        return numpy.array(indices_results).reshape(1, -1).astype(numpy.int)


class Concept2Word:
    """
    Map concept to words.
       backbone: Backbone used in CLIP or FRCNN (to use object centric annotation), currently only ResNet-50 (backbone = "RN") and Faster-RCNN (backbone="FRCNN") is supported. To use the clip
       model without alignment of activation maps, use backbone = "PlainRN".
       crop: True for using cropping to find potential words; False for using the whole example image to find potential words.
       n_potential_words: The number of potential words for each example image.
       topk: Top-k words are selected for one concept.
       device: GPU or CPU (not recommended)
       frcnn_labels_path: Path with the class and attribute name json files for the FRCNN backend
       frcnn_model_file: Trained PyTorch FRCNN model to use.
       print_word_each : True for showing words found out for each example.
    """ 
    def __init__(self, 
         backbone = 'RN', 
         crop = False,
         n_potential_words = 50,
         word_list = ['This is an image'],
         topk = 5,
         device = "cuda:0" if torch.cuda.is_available() else "cpu",
         frcnn_labels_path = "data/json/",
         frcnn_model_file = "output/frcnn/model_final.pth",
         print_word_each = False):

        self.backbone = backbone
        self.crop = crop
        self.n_potential_words = n_potential_words
        self.topk = topk
        self.word_list = word_list # Contains either a list of words, or a int -> word map for FRCNN
        self.device = device
        self.print = print_word_each
        self.frcnn_labels_path = frcnn_labels_path

        if self.backbone == 'RN' or self.backbone == 'PlainRN' :
            self.model, self.preprocess = clip.load("RN50", device=device, jit=False)
            self.model.eval()
            self.all_text_tensors = self.word2tensor(self.word_list) 
            self.target_layer = 'layer4'
        elif self.backbone == "FRCNN":
            self.predictor, self.word_list = init_frcnn(self.frcnn_labels_path, frcnn_model_file, device=self.device)
        else:
            print('backbone not found, use ResNet50 instead')
            self.model, self.preprocess = clip.load("RN50", device=device, jit=False)
            self.model.eval()
            self.all_text_tensors = self.word2tensor(self.word_list)
            self.target_layer = 'layer4'

    def word2tensor(self, words_list):
        """
        Input:
            words_list: A list of words (e.g. 10k words)
        Return:
            all_text_tensors: A tensor of word embeddings from CLIP
        """
        text_tensor = []
        batch_size = 256
        for i in range((len(words_list)//batch_size)+1):
            words = words_list[i*batch_size:(i+1)*batch_size]
            text = clip.tokenize(words).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_tensor.append(text_features)
        all_text_tensors = torch.cat(text_tensor, dim=0)
        all_text_tensors = all_text_tensors / all_text_tensors.norm(p=2, dim=1,keepdim=True)
        return all_text_tensors

    def find_closest_words(self, all_text_tensors, features, N):
        """
        Input:
            all_text_tensor: A tensor of word embeddings from CLIP
            features: A tensor of image feature from CLIP
            N: the number of words
        Return:
            ind: the index of N found out words (in the tensor of word embeddings)

        """
        # Find the closest words.
        #print(all_text_tensors.shape, features.shape)
        prods = all_text_tensors @ features.t()
        corresponding_words = []

        val, ind = torch.topk(prods, N, dim=0)
        # print(ind.shape) # shape [N,1]
        ind = ind.squeeze().cpu().numpy()
        return ind


    def crop_operation(self, concept_img_list, concept_map_list):
        """
        Input:
            concept_img_list: A list of image examples from one concept
            concept_map_list: A list of concept activation maps from one concept
        Return:
            cropped_imgs_list: A list of cropped images based on concept activation maps
        """
        concept_map_t = torch.stack(concept_map_list, dim=0)
        N_example = len(concept_map_list)
        mean_pixel = (torch.mean(torch.sum(torch.sum((concept_map_t>0.5).type(torch.FloatTensor),dim=-1),dim=-1),dim=0))
        mean_pixel = max(mean_pixel, torch.tensor(0.1*concept_map_t.size(2)*concept_map_t.size(3)))
        # print("Mean number of pixels in %s is %.4f" %(m, mean_pixel))

        crop = Window_Crop(mean_pixel, self.device, concept_map_list[0].size(-1), concept_img_list[0].size(-1))
        proposalN, coordinates_cat, coordinates_feat_cat = crop.get_information()

        # cropped_imgs = torch.zeros([N_example, 3, 224, 224]).to(device)  # [N, 4, 3, 224, 224]
        cropped_imgs_list = []
        cropped_concept_maps = torch.zeros([N_example, 3, 224, 224]).to(self.device)  # [N, 4, 3, 224, 224]
        proposalN_indices, proposalN_windows_scores, window_scores = crop(concept_map_t)
        
        coordinates = []
        coordinates_feat = []

        for j in range(N_example):
            concept_map = concept_map_list[j].unsqueeze(0)
            img = concept_img_list[j].unsqueeze(0)
            coord_tensor = torch.zeros((proposalN, 4), dtype=torch.int16)
            coord_feat_tensor = torch.zeros((proposalN, 4), dtype=torch.int16)
            for k in range(proposalN):
                [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[j]]
                [x0_, y0_, x1_, y1_] = coordinates_feat_cat[proposalN_indices[j]]
                coord_tensor[k,:] = torch.as_tensor([x0, y0, x1, y1])
                coord_feat_tensor[k,:] = torch.as_tensor([x0_, y0_, x1_, y1_])
                
                cropped_img = torch.nn.functional.interpolate(img[:,:, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                            mode='bilinear',
                                                            align_corners=True)

                cropped_imgs_list.append(cropped_img.squeeze(0))
                cropped_concept_maps[j] = torch.nn.functional.interpolate(concept_map[:,:, x0_:(x1_ + 1), y0_:(y1_ + 1)], size=(224, 224),
                                                            mode='bilinear',
                                                            align_corners=True)
                
                coordinates.append(coord_tensor)
                coordinates_feat.append(coord_feat_tensor)
        return cropped_imgs_list

    def get_words_and_scores(self, concept_img_list, concept_map_list):
        """
        The acual annotation function, that returns the meanings from the concept examples.
        Input:
            concept_img_list: A list of example images from one concept
            concept_map_list: A list of concept activation maps from one concept
        Return:
            word_dict_list: A list of k dicionaries. (k is given by topk). 
                            Each dict: {'word': word found out for the concept, 'score': MSE loss between gradcam for the word and concept activation map}.
        """
        if self.backbone == "RN" or self.backbone == "PlainRN":
            potential_w_list = []
            if self.crop:
                cropped_img_list = self.crop_operation(concept_img_list, concept_map_list)
                for img in cropped_img_list:
                    test_image = transforms.ToPILImage()(img)
                    input_img = self.preprocess(test_image)
                    with torch.no_grad():
                        image_features = self.model.encode_image(input_img.unsqueeze(0).to(self.device))
                        image_features = image_features / image_features.norm(p=2, dim=1,keepdim=True)
                        potential_w = self.find_closest_words(self.all_text_tensors, image_features, self.n_potential_words)
                    potential_w_list.extend(potential_w)
            else:
                for img in concept_img_list:
                    test_image = transforms.ToPILImage()(img)
                    input_img = self.preprocess(test_image)
                    with torch.no_grad():
                        image_features = self.model.encode_image(input_img.unsqueeze(0).to(self.device))
                        image_features = image_features / image_features.norm(p=2, dim=1,keepdim=True)
                        potential_w = self.find_closest_words(self.all_text_tensors,image_features, self.n_potential_words)
                    potential_w_list.extend(potential_w)
            potential_w_list = list(set(potential_w_list))
            if self.print:
                print(len(potential_w_list), [self.word_list[i] for i in potential_w_list])

            saliency_size = concept_map_list[0].size(-1)

            test_image = [transforms.ToPILImage()(img) for img in concept_img_list]
            img_tensor = [self.preprocess(img) for img in test_image]
            img_tensor = torch.stack(img_tensor, dim=0) # [N_example, 3, inputsize, inputsize])
        # map_tensor = [(m-torch.min(m))/(torch.max(m)-torch.min(m)) for m in concept_map_list] ### normalized already
    
        if self.backbone == "RN":
            map_tensor = torch.stack(concept_map_list, dim=0) # [N_example, 1, mapsize, mapsize]

            gcam = GradCam(model=self.model.visual, candidate_layers=self.target_layer)
            image_features = self.model.encode_image(img_tensor.to(self.device))
            image_features = image_features / image_features.norm(p=2, dim=1,keepdim=True) #[N_example, 1024]

            p_text_tensors = [self.all_text_tensors[idx] for idx in potential_w_list]
            p_text_tensors = torch.stack(p_text_tensors, dim=0)
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ p_text_tensors.t()
            probs = logits_per_image.softmax(dim=-1)  #[N_example, len(potential_w_list)]

            gt_mask = map_tensor.repeat(1,probs.size(1),1,1).to(self.device)
            masks_list = []
            for viz_w in range(probs.size(1)):
                labels = torch.zeros((probs.size()))
                labels[:,viz_w] = 1.0
                _, gcam_out, one_hot = compute_gradCAM(probs, labels, gcam, False, self.target_layer)
                gcam_mask = get_mask(gcam_out) # [N_example, 1, 7, 7]                  
                gcam_mask = torch.nn.functional.interpolate(gcam_mask, size=(saliency_size,saliency_size), mode='nearest')
                gcam_mask[gcam_mask != gcam_mask] = -1.0 # get rid of nan
                masks_list.append(gcam_mask)
            masks = torch.cat(masks_list, dim=1).detach() # [N_example, len(potential_w_list), mapsize, mapsize]
            losses = torch.nn.MSELoss(reduce=False)(masks,gt_mask).sum(dim=-1).sum(dim=-1)
            losses = losses.mean(dim=0)

            scores, w_id = torch.topk(losses, self.topk, largest=False, dim=0)

            # store words and scores(losses)
            word_dict_list = []
            for t in range(self.topk):
                word_dict = {}
                word_dict['word'] = self.word_list[potential_w_list[w_id[t]]]
                word_dict['score'] = scores[t].cpu().item()
                word_dict_list.append(word_dict)
            return word_dict_list
        elif self.backbone == "PlainRN":
            # map_tensor = [(m-torch.min(m))/(torch.max(m)-torch.min(m)) for m in concept_map_list] ### normalized already

            image_features = self.model.encode_image(img_tensor.to(self.device))
            image_features = image_features / image_features.norm(p=2, dim=1,keepdim=True) #[N_example, 1024]

            p_text_tensors = [self.all_text_tensors[idx] for idx in potential_w_list]
            p_text_tensors = torch.stack(p_text_tensors, dim=0)
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ p_text_tensors.t()
            probs = logits_per_image.softmax(dim=-1)  #[N_example, len(potential_w_list)]
            words_scores = probs.sum(axis=0)
            scores, w_id = torch.topk(words_scores, self.topk, largest=True, dim=0)
            word_dict_list = []
            for t in range(self.topk):
                word_dict = {}
                word_dict['word'] = self.word_list[potential_w_list[w_id[t]]]
                word_dict['score'] = scores[t].cpu().item()
                word_dict_list.append(word_dict)
            return word_dict_list
        elif self.backbone == "FRCNN":
            class_scores = {}
            for exemplar, act_map in zip(concept_img_list, concept_map_list):
                npimg = exemplar.transpose(0,1).transpose(1,2).numpy()*255.0
                npimg = npimg.astype(np.uint8)
                outputs = self.predictor(npimg[:, :, ::-1]) 
                discovered_instances = outputs["instances"].to("cpu")
                #fill_common_dicts(discovered_instances, common_things, common_attributes, combinations_pres)
                example_scores = compute_overlap_score(discovered_instances, act_map)
                #print(example_scores)
                class_scores = fuse_scores(class_scores, example_scores)

            class_scores = {sk: sv for sk, sv in sorted(class_scores.items(), key=lambda v: v[1], reverse=True)} # Sort by score
            top_k = 0
            score_list = []
            for k, v in class_scores.items():
                score_list.append({"word": self.word_list[k].split(".")[0], "score": v})
                if top_k >= self.topk:
                    break
                top_k += 1
            return score_list
        else:
            print("Error: Unknown backbone.")
            return {}

    def annotate(self, input_concept: Concept):
        """ Simple interfact for the annotation, that takes a Concept Object and returns the corresponding meaning
            (a list of word, score tuples.)
            :param input_concept: The Concept object, for which the meaning should be computed.
            :return: The meaning. A list of words-score tuples, with the best-fitting words found for this concept.
        """
        concept_map_list=[]
        concept_img_list=[]
        for exemplar in input_concept:
            concept_map = exemplar.saliency_map.view(-1, 14, 14)
            concept_map = (concept_map-torch.min(concept_map))/(torch.max(concept_map)-torch.min(concept_map))
            concept_map_list.append(concept_map)
            img = exemplar.image_tensor
            concept_img_list.append(img)
        word_and_score = self.get_words_and_scores(concept_img_list, concept_map_list)
        return word_and_score

class ObjectCentricAnnotation(Concept2Word):
    def __init__(self, device="cuda", topk=5, frcnn_labels_path = "data/json/", frcnn_model_file = "output/frcnn/model_final.pth"):
        """ 
        The Object-Centric annotation module relying on FRCNN.
        Arguments:
            device: Device  to run the model on.
            topk: The number of words to return.
            frcnn_labels_path: Path to folder where the frcnn label and attribute lists can be found (They are stored in data/json in the repo)
            frcnn_model_file: Path of the final FRCNN model.
        """
        super(ObjectCentricAnnotation, self).__init__(backbone="FRCNN",
                device=device, topk = topk, frcnn_labels_path=frcnn_labels_path, frcnn_model_file=frcnn_model_file)

class JointVisionLanguageAnnotation(Concept2Word):
    def __init__(self, word_list, device="cuda", topk=5,  saliency_alignment=True, crop=False, n_potential_words = 50):
        """ 
        The Joint-Vision-Language Annotation module relying on CLIP.
        Arguments:
            device: Device to run the model on.
            topk: The number of words to return.
            word_list: The vocabulary to use. A list of strings.
            saliency_alignment: Run our devised method, to compute GradCAM activations for the words and improve the scores by matching the concept 
                activation maps with the saliency maps of the words. A number of candidate words are preselected using the plain CLIP scores,
                to increase computational efficiency.
            crop: Another variation of the approach where the concept saliency map is used to crop out the important part of the image to better align the 
                clip scores in this way. However, we did not observe significant improvement with this method.
            n_potential_words: Number of candidate words that are preselected for the saliency alignment step. Only used if saliency_alignment = True.
        """
        super(JointVisionLanguageAnnotation, self).__init__(backbone=("RN" if saliency_alignment else "PlainRN"),
               word_list=word_list, device=device, topk = topk, crop=crop, n_potential_words=n_potential_words)