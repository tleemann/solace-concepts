import torch
import typing as tp
import numpy as np
import matplotlib.cm as cmap
from matplotlib.colors import ListedColormap
import pickle
from os import path, listdir

__all__ = ["Concept", "ConceptExemplar", "read_concepts", "get_concept_count"]

# Create an own colormap
viridis = cmap.get_cmap('jet')
viridis = viridis(np.linspace(0,1,40)) # [40, 4] array
thres = 0.4
# Set alpha value to 0 for the values after the threshold.
viridis[:int(thres*40),3]=0.0
newcmp = ListedColormap(viridis)

class ConceptExemplar:
    """ An example image, that highly activates the corresponding concept """
    def __init__(self, image_tensor: torch.FloatTensor, saliency_map: torch.FloatTensor, path: str, image_id: int, class_id=0) -> None:
        """ An example of an image that corresponds to a concept example. """
        self.image_tensor = image_tensor    # [3, S, S] image tensor with the original input to the model 
        self.path = path                    # dataset path for reidentification purposes
        self.image_id = image_id            # Image id in the dataset (e.g. ImageFolderDataset, mainly for debugging purposes)
        self.saliency_map = saliency_map    # [N, N] tensor with saliency map values (cos similarities between -1 and 1 (highly present)) 
                                            # The utility functions assumpe S = 32*N, e.g. S=448, N=7.
        self.class_id = class_id            # class id of example in the corresponding dataset


    def to_numpy_image(self):
        """ Return the channel last representation, that is useful 
        for showing the contents with matplotlib.pyplot.imshow"""
        return self.image_tensor.transpose(0,1).transpose(1,2)

    def to_numpy_image_with_overlay(self):
        """ Return the numpy representation of the image with the 
            saliency map as overlay. """
        
        my_colors = torch.tensor(newcmp(self.saliency_map.flatten()/torch.max(self.saliency_map))\
            .reshape(self.saliency_map.size(0), self.saliency_map.size(1), -1), dtype=float)

        my_colors = my_colors.transpose(2,1).transpose(1,0)
        #print(my_colors.shape)
        up_sal = torch.nn.functional.interpolate(my_colors.unsqueeze(0), size=(self.image_tensor.size(1), self.image_tensor.size(2)), mode='nearest').squeeze(0)
        
        alpha = 0.2*up_sal[3,:,:].reshape(1, self.image_tensor.size(1), self.image_tensor.size(2))

        imgs = (1.0-alpha)*self.image_tensor +  alpha*up_sal[:3,:,:]
        #imgs = imgs[:,::2,::2]
        return imgs.transpose(0,1).transpose(1,2)

class Concept:
    """ A set of images that correspond to a certain concept. """
    def __init__(self, exemplars: tp.List[ConceptExemplar], concept_id: int) -> None:
        self.concept_id = concept_id    # an ID
        self.exemplars = exemplars      # list of concept exemplars
        self.source_file = ""           # File name of source concept
        self.source_id = 0              # Concept ID in the source file
        self.wordslist = []             # List of determined words, words are stored as tuples of word, and algorithm which yielded them, e.g. ("grass", [0])

    def __getitem__(self, i: int) -> ConceptExemplar:
        """ Return the ith example. """
        return self.exemplars[i]
    
    def __len__(self):
        return len(self.exemplars)

def read_concepts(pathname: str, load_id: int=-1) -> tp.List[Concept]:
    """ Read concepts out of a single file or a folder 
        that contains a seperate pickle file for each concept.
        load_id: Set to a specific value, if you just want to load a single concept.
                 set to -1 to load all concepts.
    """

    if path.isdir(pathname):
        # Read directory file by file.
        eval_concepts = []
        if load_id == -1:
            for f in listdir(pathname):
                fname = path.join(pathname, f)
                if path.isfile(fname) and fname.endswith(".pickle"):
                    with open(fname, "rb") as f:
                        eval_concepts.append(pickle.load(f))
            eval_concepts = sorted(eval_concepts, key=lambda v: v.source_id, reverse=False)
            return eval_concepts
        else:
            fname = path.join(pathname, f"concept_{load_id}.pickle")
            with open(fname, "rb") as f:
                eval_concepts.append(pickle.load(f))
            return eval_concepts
    else:
        # Unpickle single file
        with open(pathname, "rb") as f:
            eval_concepts = pickle.load(f)
        if id == -1:
            return eval_concepts
        else:
            return [eval_concepts[load_id]]

def get_concept_count(pathname: str) -> int:
    """ Only get the concept count. For a single file, this is as expensive as reading the whole file,
    but for paths it will be faster. """
    if path.isdir(pathname):
        # Read directory file by file.
        num_concepts = 0
        eval_concepts = []
        for f in listdir(pathname):
            fname = path.join(pathname, f)
            if path.isfile(fname) and fname.endswith(".pickle"):
                num_concepts += 1
        return num_concepts
    else:
        # Unpickle single file
        with open(pathname, "rb") as f:
            eval_concepts = pickle.load(f)
    # Sort by source Ids.
    return len(eval_concepts)